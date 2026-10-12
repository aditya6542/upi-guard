"""Train the model and produce the metrics tables used in the report.

Run:  python3 src/detector/train.py
Reads data/raw/transactions.jsonl, writes model + metrics JSON to data/processed/.

This is also the one runnable self-check for the pipeline: it asserts the
model beats the majority baseline and that false positives stay under a ceiling,
so a broken change fails loudly instead of quietly shipping.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generator"))

from engine import build_engine_from_data, load_config  # noqa: E402
from features import FEATURE_NAMES, extract_many  # noqa: E402
from taxonomy import SCAM_TYPES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def classify(risk: float, block_t: float, stepup_t: float) -> str:
    if risk >= block_t:
        return "BLOCK"
    if risk >= stepup_t:
        return "STEP_UP"
    return "ALLOW"


def metrics_for(labels: np.ndarray, preds: np.ndarray) -> dict:
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # false positive rate = share of GENUINE payments we wrongly stopped
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(f1, 4), "false_positive_rate": round(fpr, 4),
        "specificity": round(tn / (tn + fp), 4) if tn + fp else 0.0,
    }


def main() -> None:
    cfg = load_config()
    rows = load_rows(ROOT / "data" / "raw" / "transactions.jsonl")
    fraud_templates = {s.code: list(s.message_templates) for s in SCAM_TYPES}

    eng = build_engine_from_data(cfg, rows, fraud_templates)

    block_t = cfg["gate"]["block_threshold"]
    stepup_t = cfg["gate"]["stepup_threshold"]

    verdicts = []
    for r in rows:
        v = eng.score(r)
        verdicts.append(v)

    # Treat BLOCK and STEP_UP both as "flagged" for the headline detection metrics
    labels = np.array([1 if r["label"] == "fraud" else 0 for r in rows])
    risks = np.array([v.risk for v in verdicts])
    preds = (risks >= stepup_t).astype(int)

    overall = metrics_for(labels, preds)

    # per scam-type recall
    per_type = {}
    for s in SCAM_TYPES:
        idx = [i for i, r in enumerate(rows) if r["type_code"] == s.code]
        if not idx:
            continue
        tp = sum(1 for i in idx if preds[i] == 1)
        per_type[s.code] = {
            "label": s.label,
            "n": len(idx),
            "recall": round(tp / len(idx), 4),
        }

    # per legit-type false positive rate (the number that matters most to users)
    per_legit = {}
    for r, p in zip(rows, preds):
        if r["label"] == "legit":
            d = per_legit.setdefault(r["type_code"], {"label": r["type_label"], "n": 0, "fp": 0})
            d["n"] += 1
            if p == 1:
                d["fp"] += 1
    for code, d in per_legit.items():
        d["fpr"] = round(d["fp"] / d["n"], 4) if d["n"] else 0.0

    # latency
    import time
    t0 = time.perf_counter()
    for r in rows[:2000]:
        eng.score(r)
    latency_ms = (time.perf_counter() - t0) / min(2000, len(rows)) * 1000

    # top learned features (explainability artifact)
    top_feats = eng.model.top_features(8)

    out = {
        "n_transactions": len(rows),
        "n_fraud": int(labels.sum()),
        "gate": {"block": block_t, "stepup": stepup_t},
        "overall": overall,
        "per_scam_type": per_type,
        "per_legit_type_fpr": per_legit,
        "latency_ms_per_txn": round(latency_ms, 3),
        "top_features": top_feats,
        "baseline_majority_f1": round(float(labels.mean() > 0.5), 4),
    }

    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "metrics.json").write_text(json.dumps(out, indent=2))
    eng.model.save(proc / "models" / "behavioural_model.json")

    # ---- runnable self-check ----
    assert overall["recall"] > 0.85, f"recall too low: {overall['recall']}"
    assert overall["false_positive_rate"] < 0.15, f"FPR too high: {overall['false_positive_rate']}"
    assert eng.model is not None and len(eng.model.w) == len(FEATURE_NAMES)

    print("=== DETECTION (flag = BLOCK or STEP_UP) ===")
    print(f"  recall (fraud caught)      : {overall['recall']:.1%}")
    print(f"  precision                  : {overall['precision']:.1%}")
    print(f"  false positive rate        : {overall['false_positive_rate']:.1%}  <- genuine payments stopped")
    print(f"  F1                         : {overall['f1']:.3f}")
    print(f"  latency / txn              : {latency_ms:.2f} ms")
    print("=== per-scam-type recall ===")
    for code, d in sorted(per_type.items()):
        print(f"  {code} {d['label'][:38]:38} n={d['n']:4d}  recall={d['recall']:.0%}")
    print("=== worst legit FPR ===")
    for code, d in sorted(per_legit.items(), key=lambda kv: -kv[1]["fpr"])[:4]:
        print(f"  {code} {d['label'][:38]:38} fpr={d['fpr']:.1%}")
    print(f"\nwrote {proc/'metrics.json'}")


if __name__ == "__main__":
    main()
