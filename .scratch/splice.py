"""Splice a block of text into a file between two marker lines (inclusive start,
exclusive end). Used for edits where the existing text contains control
characters that cannot be matched literally."""

import sys
from pathlib import Path

target, block_file, start_marker, end_marker = sys.argv[1:5]
path = Path(target)
text = path.read_text(encoding="utf-8")
block = Path(block_file).read_text(encoding="utf-8")

start = text.index(start_marker)
end = text.index(end_marker, start)
path.write_text(text[:start] + block + text[end:], encoding="utf-8")
print(f"spliced {len(block)} chars into {path.name} ({end - start} replaced)")
