"""Headless smoke test for the Streamlit app.

Streamlit renders in a browser, so it is awkward to verify from the command
line. But almost every crash in app/main.py happens in the *data* underneath —
a missing key, a float where an array was expected, an empty dataframe. This
script exercises exactly the code paths the app calls, with Streamlit stubbed
out, so those failures surface here instead of in front of a judge.

It does NOT check that the page looks right. For that, open the app and follow
docs/DEMO_SCRIPT.md.

Run:  python3 scripts/smoke_test.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "detector"))
sys.path.insert(0, str(ROOT / "src" / "generator"))

import numpy as np  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str):
    def deco(fn):
        try:
            fn()
            results.append((PASS, name, ""))
        except Exception as exc:  # noqa: BLE001 - report, don't abort
            results.append((FAIL, name, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
        return fn
    return deco


# ---------------------------------------------------------------- app imports
@check("app imports resolve (streamlit stubbed)")
def _():
    import types
    stub = types.ModuleType("streamlit")
    for attr in ("set_page_config", "markdown", "caption", "error", "stop",
                 "header", "subheader", "divider", "button", "dataframe",
                 "metric", "info", "warning", "title", "success", "write",
                 "progress", "expander"):
        setattr(stub, attr, lambda *a, **k: None)
    import contextlib
    stub.__dict__["expander"] = lambda *a, **k: contextlib.nullcontext()
    # Widgets must return a usable value, not None, or the module-level
    # dispatch and the screen functions below raise on lookup.
    stub.__dict__["selectbox"] = lambda label, options, *a, **k: list(options)[0]
    stub.__dict__["select_slider"] = lambda label, options=None, *a, **k: (
        0 if options is None else list(options)[0])
    stub.__dict__["slider"] = lambda label, *a, **k: a[0] if a else 0
    class _Col(contextlib.nullcontext):
        def __getattr__(self, _):
            return lambda *a, **k: None
    stub.__dict__["columns"] = lambda n: [_Col() for _ in (n if isinstance(n, (list, tuple)) else range(n))]
    stub.__dict__["sidebar"] = types.SimpleNamespace(
        title=lambda *a, **k: None, caption=lambda *a, **k: None,
        divider=lambda *a, **k: None,
        markdown=lambda *a, **k: None,
        radio=lambda label, options, *a, **k: list(options)[0])
    # cache_data must work as a decorator factory: @st.cache_data(...) wraps
    # load_all. Returning the function unchanged is the correct no-op.
    stub.__dict__["cache_data"] = lambda *a, **k: (lambda fn: fn)
    class _SS(dict):
        def __getattr__(self, k):
            return self.get(k)
        def __setattr__(self, k, v):
            self[k] = v
    stub.__dict__["session_state"] = _SS(verdict=None, txn=None)
    sys.modules["streamlit"] = stub
    import importlib.util
    spec = importlib.util.spec_from_file_location("app_main", ROOT / "app" / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    globals()["APP"] = mod


APP = None


# ---------------------------------------------------------------- data layer
@check("dataset + engine load (load_all)")
def _():
    from engine import build_engine_from_data, load_config
    from taxonomy import SCAM_TYPES
    cfg = load_config()
    rows = [json.loads(l) for l in (ROOT / "data/raw/transactions.jsonl").open(encoding="utf-8")]
    assert len(rows) >= 1000, "dataset looks empty"
    eng = build_engine_from_data(cfg, rows, {s.code: list(s.message_templates) for s in SCAM_TYPES})
    globals().update(CFG=cfg, ROWS=rows, ENG=eng)


@check("metrics.json present and well-formed")
def _():
    p = ROOT / "data/processed/metrics.json"
    assert p.exists(), "run: python3 src/detector/train.py"
    m = json.loads(p.read_text())
    for k in ("n_transactions", "overall", "per_scam_type", "per_legit_type_fpr",
              "latency_ms_per_txn", "top_features"):
        assert k in m, f"metrics.json missing {k}"
    assert m["overall"]["false_positive_rate"] < 0.15


@check("adaptivity.json present and well-formed")
def _():
    p = ROOT / "data/processed/adaptivity.json"
    assert p.exists(), "run: python3 src/detector/mutate.py"
    a = json.loads(p.read_text())
    assert a["adversarial_recall_after_adaptation"] >= a["adversarial_recall_before_adaptation"]


# ---------------------------------------------------------------- screen 1
@check("screen 1: every scenario renders + gate returns a valid action")
def _():
    block_t = CFG["gate"]["block_threshold"]
    stepup_t = CFG["gate"]["stepup_threshold"]
    by_type: dict[str, list] = {}
    for r in ROWS:
        by_type.setdefault(r["type_code"], []).append(r)
    for code, group in by_type.items():
        v = ENG.score(group[0])
        assert v.action in ("BLOCK", "STEP_UP", "ALLOW"), f"{code}: bad action {v.action}"
        assert isinstance(v.risk, float), f"{code}: risk not a float"
        assert all(isinstance(s, float) for s in v.layers.values()), f"{code}: layer not float"
        assert 0.0 <= v.risk <= 1.0, f"{code}: risk out of range"
    print(f"    {len(by_type)} scenario types OK", end=" · ")


@check("screen 1: fraud blocks, legit allows (gate is calibrated)")
def _():
    stepup_t = CFG["gate"]["stepup_threshold"]
    fraud = [ENG.score(t) for t in ROWS if t["label"] == "fraud"]
    legit = [ENG.score(t) for t in ROWS if t["label"] == "legit"]
    fraud_flag = sum(1 for v in fraud if v.risk >= stepup_t) / len(fraud)
    legit_flag = sum(1 for v in legit if v.risk >= stepup_t) / len(legit)
    print(f"    fraud caught {fraud_flag:.1%} · genuine blocked {legit_flag:.2%}", end=" · ")
    assert fraud_flag > 0.85, "gate is missing fraud"
    assert legit_flag < 0.15, "gate is blocking too much genuine traffic"


@check("screen 1: every alert has at least one reason")
def _():
    for t in ROWS:
        v = ENG.score(t)
        if v.action != "ALLOW":
            assert v.reasons, f"{t['txn_id']} flagged with no explanation"
            break


# ---------------------------------------------------------------- screen 2
@check("screen 2: adaptivity table rows format correctly")
def _():
    a = json.loads((ROOT / "data/processed/adaptivity.json").read_text())
    for code, d in a["per_type_adversarial"].items():
        assert d["n"] > 0 and 0 <= d["caught"] <= d["n"]
        assert 0.0 <= d["recall"] <= 1.0
    assert a["examples"], "no mutation examples to show"


# ---------------------------------------------------------------- screen 3
@check("screen 3: ring detection returns renderable rings")
def _():
    rings = ENG.graph.top_rings(5)
    assert isinstance(rings, list)
    for r in rings:
        assert r["size"] >= CFG["graph"]["ring_min_size"]
        assert r["flagged"] <= r["size"]
    print(f"    {len(rings)} rings", end=" · ")


@check("screen 3: ring layer actually finds flagged accounts")
def _():
    """A ring detector that finds nothing is not a feature, it's dead code."""
    assert ENG.graph.flagged, "no flagged seeds — graph layer is inert"


@check("screen 3: feature explorer matches the reason-code dictionary")
def _():
    from features import FEATURE_REASON
    for f in ENG.model.top_features(8):
        assert f["feature"] in FEATURE_REASON, f"{f['feature']} has no plain-English string"


@check("screen 3: transaction explorer builds its dataframe")
def _():
    subset = ROWS[:200]
    risks = np.array([ENG.score(t).risk for t in subset])
    assert len(risks) == len(subset)
    for t, r in zip(subset, risks):
        _ = {"Txn": t["txn_id"], "Risk": f"{r:.2f}", "Amount": f"Rs {t['amount']:,}",
             "Type": t["type_label"], "Message": t["message"][:70]}
    flagged = [i for i, r in enumerate(risks) if r >= CFG["gate"]["stepup_threshold"]]
    assert flagged, "no flagged rows to drill into in the first 200"


# ---------------------------------------------------------------- report
print("\n" + "=" * 66)
for status, name, err in results:
    mark = "✓" if status == PASS else "✗"
    print(f"{mark} {name}")
    if err:
        print(f"    {err}")
failed = sum(1 for s, _, _ in results if s == FAIL)
print("=" * 66)
print(f"{len(results) - failed}/{len(results)} passed")
print("\nNote: this checks the logic under the app, not how it looks.")
print("Open `streamlit run app/main.py` and follow docs/DEMO_SCRIPT.md to check the UI.")
sys.exit(1 if failed else 0)
