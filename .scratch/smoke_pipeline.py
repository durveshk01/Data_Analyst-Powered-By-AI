import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.ai.tools import ToolContext, execute_tool  # noqa: E402
from app.demo.generator import DEMO_QUESTIONS, generate_demo_dataframe  # noqa: E402
from app.engine.insights import generate_insights  # noqa: E402
from app.engine.kpi import compute_kpis  # noqa: E402
from app.engine.profiling import profile_dataframe  # noqa: E402
from app.engine.quality import assess_quality  # noqa: E402
from app.engine.semantics import build_semantic_model  # noqa: E402
from app.engine.utils import unit_scope  # noqa: E402
from app.nl.intent import parse_question  # noqa: E402
from app.nl.narrator import narrate  # noqa: E402
from app.nl.planner import plan  # noqa: E402

rows = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
t0 = time.time()
df = generate_demo_dataframe(rows=rows, seed=20260101)
profile = profile_dataframe(df)
sem = build_semantic_model(profile)
units = {c.name: c.unit for c in profile.columns}
print(f"pipeline input {df.shape} in {time.time() - t0:.1f}s")
print("units:", {k: v for k, v in units.items() if v != "none"})
print(f"grain native={sem.primary_time_grain} reporting={sem.reporting_grain}")
print("kpi candidates:", [(k["column"], k["agg"]) for k in sem.kpi_candidates])

with unit_scope(units):
    t = time.time()
    q = assess_quality(df, profile)
    print(f"\nQUALITY {q.score} {q.grade} ({time.time() - t:.2f}s) dims={q.dimension_scores}")
    for i in q.issues[:10]:
        print(f"  [{i.severity.value:8}] {i.title}")

    t = time.time()
    kpis = compute_kpis(df, profile=profile, semantics=sem)
    print(f"\nKPIS ({time.time() - t:.2f}s)")
    for k in kpis:
        print(f"  {k.label:22} {k.formatted:>12}  delta={k.delta_pct} good={k.is_good} {k.delta_label}")

    t = time.time()
    insights = generate_insights(df, profile=profile, semantics=sem, quality=q)
    print(f"\nINSIGHTS ({time.time() - t:.2f}s)")
    for ins in insights:
        print(f"  [{ins.severity.value:8} {ins.importance:5.1f}] {ins.kind:14} {ins.title}")
        if ins.recommendation:
            print(f"        rec: {ins.recommendation[:160]}")

    from app.engine.insights import GENERATORS, InsightContext

    gctx = InsightContext(df, profile, sem, q)
    for gen in GENERATORS:
        t = time.time()
        gen(gctx)
        print(f"    {gen.__name__:28} {time.time() - t:.2f}s")

    ctx = ToolContext(df=df, profile=profile, semantics=sem, quality=q, dataset_name="demo")
    print("\nQUESTIONS")
    for question in DEMO_QUESTIONS:
        t = time.time()
        intent = parse_question(question, sem)
        p = plan(intent, ctx)
        result = execute_tool(ctx, p.tool, p.arguments)
        answer = narrate(result, intent, notes=p.fallback_notes, suggestions=p.suggestions)
        print(f"\nQ: {question}  -> {intent.intent} ({intent.confidence}) -> {p.tool} {p.arguments}  [{time.time() - t:.2f}s]")
        print(f"   ANSWER: {answer.answer}")
        if answer.why:
            print(f"   WHY: {answer.why[:300]}")
        if answer.recommendation:
            print(f"   REC: {answer.recommendation[:200]}")
        if answer.evidence:
            print(f"   EVIDENCE[0]: {answer.evidence[0]}")
        if not answer.ok:
            print("   !!! NOT OK")
        for n in answer.notes:
            print(f"   NOTE: {n[:200]}")
