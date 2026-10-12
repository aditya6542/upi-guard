"""Behavioural model: logistic regression trained with numpy only.

A gradient-boosted tree would score marginally better on tabular data, but the
linear model is the right call here for two reasons:

1. Every weight is directly readable as a reason code. The alert can say "first
   payment to a 6-day-old payee" without a SHAP dependency.
2. It trains in under a second on CPU with no ML stack, which is what makes the
   analyst feedback loop (retrain on new labels) a live demo rather than a
   batch job.

Writes its coefficients to JSON so the dashboard can show the top drivers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from features import FEATURE_NAMES, FEATURE_REASON


class LogisticModel:
    def __init__(self, lr: float = 0.1, epochs: int = 400, l2: float = 1e-3, seed: int = 0):
        self.w = np.zeros(len(FEATURE_NAMES))
        self.b = 0.0
        self.lr, self.epochs, self.l2, self.seed = lr, epochs, l2, seed

    @staticmethod
    def _sigmoid(z):
        """Element-wise sigmoid. Accepts a scalar or an array of any shape."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticModel":
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        order = rng.permutation(n)
        for _ in range(self.epochs):
            for idx in order:
                xi, yi = X[idx], y[idx]
                # float() on the scalar result, not on a 1-element array:
                # NumPy 2.x removed that implicit conversion.
                p = self._sigmoid(float(self.w @ xi + self.b))
                err = yi - p
                self.w += self.lr * (err * xi - self.l2 * self.w)
                self.b += self.lr * err
        return self

    def score(self, x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        return float(self._sigmoid(float(self.w @ x + self.b)))

    def score_many(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.zeros(0)
        return self._sigmoid(X @ self.w + self.b)

    def reasons(self, x: np.ndarray, top_k: int = 3) -> list[dict]:
        """Contributions that pushed this transaction toward fraud (w*x, positive).

        Feature *values* are also shown so the reason is concrete, not abstract.
        """
        contribs = self.w * x
        order = np.argsort(-contribs)[:top_k]
        out = []
        for i in order:
            if contribs[i] <= 1e-6:
                break
            out.append({
                "feature": FEATURE_NAMES[int(i)],
                "text": FEATURE_REASON[FEATURE_NAMES[int(i)]],
                "contribution": float(contribs[i]),
                "value": float(x[int(i)]),
            })
        return out

    def top_features(self, k: int = 8) -> list[dict]:
        order = np.argsort(-np.abs(self.w))[:k]
        return [{"feature": FEATURE_NAMES[int(i)], "weight": float(self.w[int(i)])} for i in order]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"w": self.w.tolist(), "b": self.b,
                                 "feature_names": FEATURE_NAMES}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LogisticModel":
        d = json.loads(Path(path).read_text())
        m = cls()
        m.w = np.array(d["w"], dtype=np.float64)
        m.b = float(d["b"])
        return m
