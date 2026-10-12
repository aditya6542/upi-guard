"""Detector orchestration and risk fusion.

Four independent layers score a transaction; this module runs them, blends the
scores with configurable weights, and applies the pre-payment gate.

Fusion is a weighted sum, then a sigmoid so the output is a 0..1 probability.
That is deliberately simpler than a learned stacking model: with four layers
that mostly agree, the weighted sum is nearly as good and — crucially — each
layer's contribution stays visible in the final number, which the dashboard
shows. A black-box blend would undo the explainability the whole design is
built around.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from features import extract, extract_many
from graph import RingDetector
from model import LogisticModel
from rules import evaluate
from similarity import TfidfIndex, build_fraud_index

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    import yaml
    p = Path(path) if path else ROOT / "configs" / "config.yaml"
    return yaml.safe_load(p.read_text())


@dataclass
class LayerScore:
    name: str
    score: float
    reasons: list[dict] = field(default_factory=list)


@dataclass
class Verdict:
    txn_id: str
    risk: float
    action: str  # BLOCK | STEP_UP | ALLOW
    layers: dict[str, float]
    reasons: list[dict]  # merged, human-readable, deduped

    @property
    def reason_texts(self) -> list[str]:
        return [r["text"] for r in self.reasons]


class GuardEngine:
    """The full risk engine. Construct once, score many."""

    def __init__(self, cfg: dict | None = None, fraud_templates: dict[str, list[str]] | None = None):
        self.cfg = cfg or load_config()
        w = self.cfg.get("fusion", {}).get("weights", {})
        self.w_rules = w.get("rules", 0.34)
        self.w_script = w.get("script", 0.24)
        self.w_behav = w.get("behavioural", 0.22)
        self.w_graph = w.get("graph", 0.20)

        self.tfidf: TfidfIndex | None = None
        if fraud_templates:
            self.tfidf = build_fraud_index(fraud_templates)

        self.model: LogisticModel | None = None
        self.graph = RingDetector(self.cfg.get("graph", {}))

    # ------------------------------------------------------------ ingestion
    def fit_tfidf(self, fraud_templates: dict[str, list[str]]) -> None:
        self.tfidf = build_fraud_index(fraud_templates)

    def fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model = LogisticModel().fit(X, y)

    def ingest_graph(self, txns: list[dict], seed_accounts: set[str] | None = None) -> None:
        self.graph.ingest(txns, seed_accounts)

    def seed_from_blocks(self, txns: list[dict]) -> set[str]:
        """Build the graph seed-blocklist from the rule layer alone.

        Two-pass by necessity: seeds cannot come from the graph layer (it would
        be assuming its own answer) and must not come from the ground-truth
        label (that would be cheating). In deployment this is the set of
        accounts a previous run's gate blocked, so we replay the deterministic
        rule layer here to stand in for it.
        """
        seeds: set[str] = set()
        for t in txns:
            score, _ = evaluate(t, self.cfg)
            if score >= self.cfg.get("gate", {}).get("block_threshold", 0.72):
                acct = t.get("user_home_vpa") or t.get("payee_vpa")
                if acct:
                    seeds.add(acct)
        return seeds

    # ------------------------------------------------------------ scoring
    def _script_score(self, message: str) -> tuple[float, list[dict]]:
        if not self.tfidf:
            return 0.0, []
        sim, labels = self.tfidf.query(message)
        score = max(0.0, min(1.0, (sim - 0.25) / 0.5))  # rescale: 0.25->0, 0.75->1
        reasons = []
        if score > 0.25:
            fam = ", ".join(labels[:2]) if labels else "known scam"
            reasons.append({
                "layer": "script",
                "text": f"Message closely matches known {fam} scripts ({sim:.0%} similar).",
            })
        return score, reasons

    def _behav_score(self, txn: dict) -> tuple[float, list[dict]]:
        if not self.model:
            return 0.0, []
        x = extract(txn)
        score = self.model.score(x)
        reasons = []
        for r in self.model.reasons(x, top_k=3):
            reasons.append({"layer": "behavioural", "text": f"{r['text']}."})
        return score, reasons

    def score(self, txn: dict) -> Verdict:
        rules_score, rule_reasons = evaluate(txn, self.cfg)
        script_score, script_reasons = self._script_score(txn.get("message") or "")
        behav_score, behav_reasons = self._behav_score(txn)
        graph_score, graph_detail = self.graph.risk(txn)

        graph_reasons = []
        if graph_score > 0.3:
            if graph_detail["flagged_neighbours"]:
                n = graph_detail["flagged_neighbours"]
                graph_reasons.append({
                    "layer": "graph",
                    "text": (f"Device/phone shared with {n} previously flagged "
                             f"{'account' if n==1 else 'accounts'}."),
                })
            if graph_detail["component_size"] >= self.cfg.get("graph", {}).get("ring_min_size", 3):
                graph_reasons.append({
                    "layer": "graph",
                    "text": (f"Part of a connected ring of {graph_detail['component_size']} "
                             f"accounts acting together."),
                })

        blended = (self.w_rules * rules_score
                   + self.w_script * script_score
                   + self.w_behav * behav_score
                   + self.w_graph * graph_score)
        # push into a proper probability range; sigmoid keeps it monotone in `blended`
        risk = float(1.0 / (1.0 + np.exp(-6.0 * (blended - 0.35))))

        gate = self.cfg.get("gate", {})
        block_t = gate.get("block_threshold", 0.72)
        stepup_t = gate.get("stepup_threshold", 0.38)
        if risk >= block_t:
            action = "BLOCK"
        elif risk >= stepup_t:
            action = "STEP_UP"
        else:
            action = "ALLOW"

        # Dedupe reasons by text, keep priority order (rule > graph > script > behavioural)
        reasons = []
        seen = set()
        for r in (rule_reasons + graph_reasons + script_reasons + behav_reasons):
            t = r.get("detail") or r.get("text")
            if t not in seen:
                seen.add(t)
                reasons.append({"text": t, "layer": r.get("layer", "rule")})

        return Verdict(
            txn_id=txn.get("txn_id", "?"),
            risk=risk,
            action=action,
            layers={
                "rules": rules_score,
                "script": script_score,
                "behavioural": behav_score,
                "graph": graph_score,
            },
            reasons=reasons,
        )


def build_engine_from_data(cfg: dict, txns: list[dict], fraud_templates: dict) -> GuardEngine:
    """Convenience: index TF-IDF, train model, build graph, all from one dataset.

    The graph is seeded from the rule layer's blocks, never from the ground-truth
    label — see `seed_from_blocks` for why that distinction matters for the
    honesty of the reported numbers.
    """
    eng = GuardEngine(cfg)
    eng.fit_tfidf(fraud_templates)

    X = extract_many(txns)
    y = np.array([1.0 if t["label"] == "fraud" else 0.0 for t in txns])
    eng.fit_model(X, y)
    seeds = eng.seed_from_blocks(txns)
    eng.ingest_graph(txns, seed_accounts=seeds)
    return eng
