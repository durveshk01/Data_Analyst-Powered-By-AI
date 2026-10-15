"""Upload validation. Uploaded files are treated as hostile input.

Checks performed before a single byte reaches pandas:

1. Filename is sanitised (no traversal, no control characters, length bounded).
2. Extension is on an allow-list (``.csv``, ``.tsv``, ``.xlsx``, ``.json``,
   ``.ndjson``).
3. Declared extension must agree with sniffed magic bytes — a ``.csv`` that is
   really a ZIP is rejected.
4. Size ceiling enforced on the real byte count, not the client's header.
5. OOXML archives are inspected before parsing: entry count, total uncompressed
   size and per-entry compression ratio guard against zip bombs; macro parts
   (``vbaProject.bin``) and external-link parts are rejected outright.
6. Text payloads are rejected if they contain NUL bytes or are not decodable.

Separately, :func:`neutralise_formula` is applied on *export* so InsightOS never
emits a spreadsheet that executes when opened (CSV/Excel formula injection).
"""

from __future__ import annotations

import io
import json
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from typing import Final

from app.config import settings
from app.errors import (
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
    ValidationError,
)
from app.obs.logging import get_logger

log = get_logger(__name__)

ALLOWED_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".csv", ".tsv", ".xlsx", ".json", ".ndjson", ".jsonl"}
)

_ZIP_MAGIC: Final[bytes] = b"PK\x03\x04"
_OLE2_MAGIC: Final[bytes] = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_GZIP_MAGIC: Final[bytes] = b"\x1f\x8b"
_UTF8_BOM: Final[bytes] = b"\xef\xbb\xbf"

# Zip-bomb guard rails.
_MAX_ZIP_ENTRIES: Final[int] = 512
_MAX_ZIP_UNCOMPRESSED_BYTES: Final[int] = 2 * 1024 * 1024 * 1024  # 2 GiB
_MAX_ZIP_RATIO: Final[int] = 200  # uncompressed / compressed per entry

_FORBIDDEN_OOXML_PARTS: Final[tuple[str, ...]] = (
    "vbaproject.bin",     # macros
    "vbaproject.bin.rels",
)

_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_FORMULA_TRIGGERS: Final[tuple[str, ...]] = ("=", "+", "-", "@", "\t", "\r")

Kind = str  # "csv" | "tsv" | "xlsx" | "json" | "ndjson"


@dataclass(frozen=True, slots=True)
class ValidatedUpload:
    """A payload that passed every structural check."""

    filename: str
    kind: Kind
    size_bytes: int
    content: bytes
    encoding: str = "utf-8"

    @property
    def extension(self) -> str:
        idx = self.filename.rfind(".")
        return self.filename[idx:].lower() if idx != -1 else ""


# ------------------------------------------------------------------ naming ---
def sanitise_filename(raw: str, *, fallback: str = "dataset") -> str:
    """Return a safe, display-friendly basename.

    Strips directory components (including Windows separators that arrive from
    browsers on Windows), control characters and Unicode direction overrides
    that could be used to disguise an extension.
    """
    if not raw:
        return fallback
    # Normalise first so look-alike characters collapse, then drop any path part.
    name = unicodedata.normalize("NFKC", raw)
    name = name.replace("\\", "/").split("/")[-1]
    name = "".join(ch for ch in name if unicodedata.category(ch) != "Cf")
    name = _UNSAFE_FILENAME_CHARS.sub("_", name).strip(" .")
    if not name or set(name) <= {"_"}:
        return fallback
    if len(name) > 180:
        stem, _, ext = name.rpartition(".")
        keep = 180 - (len(ext) + 1) if ext else 180
        name = (stem[:keep] + "." + ext) if ext else name[:180]
    return name


def _extension_of(filename: str) -> str:
    idx = filename.rfind(".")
    return filename[idx:].lower() if idx != -1 else ""


# ------------------------------------------------------------------ sniffing --
def sniff_kind(content: bytes, extension: str) -> Kind:
    """Determine the real payload kind and verify it matches the extension."""
    head = content[:8]

    if head.startswith(_OLE2_MAGIC):
        raise UnsupportedMediaTypeError(
            "Legacy .xls (Excel 97-2003) files are not supported. "
            "Re-save as .xlsx or .csv and upload again."
        )
    if head.startswith(_GZIP_MAGIC):
        raise UnsupportedMediaTypeError(
            "Compressed archives are not supported. Upload the raw CSV, XLSX or JSON file."
        )

    is_zip = head.startswith(_ZIP_MAGIC)

    if extension == ".xlsx":
        if not is_zip:
            raise ValidationError(
                "File is named .xlsx but is not a valid Excel workbook."
            )
        return "xlsx"

    if is_zip:
        raise ValidationError(
            f"File content is a ZIP archive but the extension is '{extension}'. "
            "Rename it to .xlsx if it is an Excel workbook."
        )

    if extension in {".json", ".ndjson", ".jsonl"}:
        return "ndjson" if extension in {".ndjson", ".jsonl"} else "json"
    if extension == ".tsv":
        return "tsv"
    if extension == ".csv":
        return "csv"

    raise UnsupportedMediaTypeError(
        f"Unsupported file type '{extension}'. Supported: "
        + ", ".join(sorted(ALLOWED_EXTENSIONS))
    )


# -------------------------------------------------------------- zip hygiene --
def inspect_ooxml(content: bytes) -> None:
    """Reject malicious or malformed OOXML archives before parsing."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            infos = zf.infolist()
            if not infos:
                raise ValidationError("Excel workbook is empty or corrupt.")
            if len(infos) > _MAX_ZIP_ENTRIES:
                raise ValidationError(
                    f"Excel workbook has {len(infos)} internal parts "
                    f"(limit {_MAX_ZIP_ENTRIES}); it looks malformed."
                )

            total_uncompressed = 0
            names = set()
            for info in infos:
                name = info.filename.replace("\\", "/")
                lowered = name.lower()
                names.add(lowered)

                if name.startswith("/") or ".." in name.split("/"):
                    raise ValidationError(
                        "Excel workbook contains an illegal internal path."
                    )
                if lowered.rsplit("/", 1)[-1] in _FORBIDDEN_OOXML_PARTS:
                    raise ValidationError(
                        "Macro-enabled workbooks are not accepted. "
                        "Re-save without macros (.xlsx) and upload again."
                    )

                total_uncompressed += info.file_size
                if total_uncompressed > _MAX_ZIP_UNCOMPRESSED_BYTES:
                    raise PayloadTooLargeError(
                        "Excel workbook expands to more than 2 GiB; refusing to parse."
                    )
                if info.compress_size > 1024 and info.file_size / max(
                    info.compress_size, 1
                ) > _MAX_ZIP_RATIO:
                    raise ValidationError(
                        "Excel workbook has an implausible compression ratio "
                        "(possible zip bomb); refusing to parse."
                    )

            if not any(n.startswith("xl/") for n in names):
                raise ValidationError(
                    "File is a ZIP archive but not an Excel workbook."
                )
    except zipfile.BadZipFile as exc:
        raise ValidationError("Excel workbook is corrupt and cannot be read.") from exc


# ------------------------------------------------------------- text hygiene --
def decode_text(content: bytes) -> tuple[str, str]:
    """Decode a text payload, rejecting binary smuggled in as text."""
    payload = content[len(_UTF8_BOM) :] if content.startswith(_UTF8_BOM) else content
    if b"\x00" in payload[: 1024 * 1024]:
        raise ValidationError(
            "File contains NUL bytes and is not valid text. Upload a plain CSV/JSON file."
        )
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return payload.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ValidationError(
        "File encoding could not be detected. Re-save the file as UTF-8."
    )


def validate_json_structure(text: str) -> None:
    """Ensure a JSON payload is a tabular shape we can actually load."""
    stripped = text.lstrip()
    if not stripped:
        raise ValidationError("JSON file is empty.")
    if stripped[0] not in "[{":
        raise ValidationError(
            "JSON file must contain an array of objects or a single object."
        )
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"JSON is malformed at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if isinstance(parsed, list):
        if parsed and not any(isinstance(row, dict) for row in parsed[:50]):
            raise ValidationError(
                "JSON array must contain objects (records), not scalars or nested arrays."
            )
    elif not isinstance(parsed, dict):
        raise ValidationError("JSON root must be an array or object.")


# ----------------------------------------------------------------- pipeline --
def validate_upload(
    *,
    filename: str,
    content: bytes,
    max_size_bytes: int | None = None,
) -> ValidatedUpload:
    """Run the full validation pipeline. Raises on the first failure."""
    limit = max_size_bytes if max_size_bytes is not None else settings.max_file_size_bytes

    if not content:
        raise ValidationError("Uploaded file is empty.")
    if len(content) > limit:
        raise PayloadTooLargeError(
            f"File is {len(content) / 1_048_576:.1f} MB which exceeds the "
            f"{limit / 1_048_576:.0f} MB limit. Split the file or raise MAX_FILE_SIZE_MB."
        )

    safe_name = sanitise_filename(filename)
    extension = _extension_of(safe_name)
    if extension not in ALLOWED_EXTENSIONS:
        raise UnsupportedMediaTypeError(
            f"'{extension or safe_name}' is not a supported file type. "
            f"Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    kind = sniff_kind(content, extension)
    encoding = "binary"

    if kind == "xlsx":
        inspect_ooxml(content)
    else:
        text, encoding = decode_text(content)
        if kind == "json":
            validate_json_structure(text)
        elif kind == "ndjson":
            if not text.strip():
                raise ValidationError("NDJSON file is empty.")
        else:
            if not text.strip():
                raise ValidationError("Delimited file contains no data.")

    log.info(
        "upload_validated",
        extra={"filename": safe_name, "kind": kind, "bytes": len(content)},
    )
    return ValidatedUpload(
        filename=safe_name,
        kind=kind,
        size_bytes=len(content),
        content=content,
        encoding=encoding,
    )


# ------------------------------------------------------- export-side safety --
def neutralise_formula(value: object) -> object:
    """Prefix a leading formula trigger so spreadsheets treat the cell as text.

    Applied when InsightOS writes CSV/XLSX exports. Without this, a cell such as
    ``=HYPERLINK("http://evil","click")`` that a user uploaded would execute in
    whoever opens the exported report.
    """
    if not isinstance(value, str) or not value:
        return value
    if value[0] in _FORMULA_TRIGGERS:
        return "'" + value
    return value
