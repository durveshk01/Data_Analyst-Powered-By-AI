"""Check that the planted demo scenarios are discoverable by the engine."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.ai.tools import ToolContext, execute_tool  # noqa: E402
from app.demo.generator import generate_demo_dataframe  # noqa: E402
from app.engine.profiling import profile_dataframe  # noqa: E402
from app.engine.semantics import build_semantic_model  # noqa: E402
from app.engine.utils import unit_scope  # noqa: E402

rows = int(sys.argv[1]) if len(sys.argv) > 1 else 120000
df = generate_demo_dataframe(rows=rows, seed=20260101)
profile = profile_dataframe(df)
sem = build_semantic_model(profile)
ctx = ToolContext(df=df, profile=profile, semantics=sem)
print("rows", len(df))

with unit_scope({c.name: c.unit for c in profile.columns}):
    r = execute_tool(ctx, "explain_change", {"measure": "revenue", "period_b": "September"})
    res = r.result or {}
    print("explain ok:", r.ok, r.error)
    print("dimension:", res.get("dimension"), "scores:", res.get("dimension_scores"))
    for d in (res.get("drivers") or [])[:5]:
        print("  ", d["value"], round(d["change_abs"] or 0), d["contribution_pct"])

    r = execute_tool(ctx, "detect_anomalies", {"column": "revenue"})
    res = r.result or {}
    print("anomaly method:", res.get("method"), "count:", res.get("count"))
    labels = [a["label"] for a in res.get("anomalies", [])]
    for planted in ("2026-09-17", "2026-07-11", "2026-03-04", "2026-03-05"):
        print("  planted", planted, "found" if planted in labels else "MISSING")
    for a in res.get("anomalies", [])[:10]:
        print("  ", a["label"], a["deviation_pct"], a["score"])
