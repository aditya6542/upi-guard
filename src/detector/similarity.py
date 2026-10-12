"""Script similarity engine.

Fraud is a script: the same sentence over and over with small edits. This layer
answers "how close is this message to the fraud corpus, and which known scam
does it resemble".

Implemented as TF-IDF cosine similarity in pure Python. No sklearn, because the
risk engine should run with nothing installed, and because a hand-rolled TF-IDF
keeps the "why" (the matched template) attached to the score, which is the
whole point of the demo. Tokenizer handles the Hinglish/typo text that appears
in real scam SMS by also emitting character 3-grams, so "verfy" and "verify"
still share tokens.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9@.]+")


def _tokens(text: str) -> list[str]:
    words = _TOKEN_RE.findall(text.lower())
    grams = []
    for w in words:
        if len(w) >= 4:
            # character 3-grams: typo tolerance without a stemmer
            grams.extend(w[i:i + 3] for i in range(len(w) - 2))
    return words + grams


class TfidfIndex:
    """Small in-memory TF-IDF index over known messages."""

    def __init__(self) -> None:
        self.docs: list[list[str]] = []          # token lists
        self.doc_freq: Counter[str] = Counter()  # df per term
        self.labels: list[str] = []              # source label per doc (e.g. F01)
        self.n = 0
        # Lazily-cached unit vectors. Scoring a transaction re-weights every
        # document by the IDF computed from `doc_freq`, so the vectors are
        # invariant between queries and can be built once. Without this the
        # corpus scan is quadratic and a per-transaction budget disappears.
        self._vecs: list[dict[str, float]] | None = None

    def add(self, text: str, label: str = "") -> None:
        toks = _tokens(text)
        self.docs.append(toks)
        self.labels.append(label)
        self.doc_freq.update(set(toks))
        self.n += 1
        self._vecs = None

    def _idf(self, term: str) -> float:
        # smoothed idf; unseen terms get max weight
        return math.log((self.n + 1) / (self.doc_freq.get(term, 0) + 1)) + 1.0

    def _vector(self, toks: list[str]) -> dict[str, float]:
        if not toks:
            return {}
        tf = Counter(toks)
        vec = {t: (c / len(toks)) * self._idf(t) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def _ensure_vectors(self) -> list[dict[str, float]]:
        if self._vecs is None:
            self._vecs = [self._vector(doc) for doc in self.docs]
        return self._vecs

    def query(self, text: str, top_k: int = 1) -> tuple[float, list[str]]:
        """Return (best_similarity, labels_of_top_k)."""
        q = self._vector(_tokens(text))
        if not q:
            return 0.0, []
        scores: list[tuple[float, str]] = []
        for d, label in zip(self._ensure_vectors(), self.labels):
            if not d:
                continue
            # dot product of unit-norm sparse vectors
            small, large = (q, d) if len(q) <= len(d) else (d, q)
            sim = sum(v * large.get(t, 0.0) for t, v in small.items())
            scores.append((sim, label))
        if not scores:
            return 0.0, []
        scores.sort(reverse=True)
        best = scores[0][0]
        seen: list[str] = []
        for _sim, label in scores[:top_k]:
            if label and label not in seen:
                seen.append(label)
        return float(best), seen


def build_fraud_index(templates_by_code: dict[str, list[str]]) -> TfidfIndex:
    """Index the scam message templates (their code = which scam family)."""
    idx = TfidfIndex()
    for code, templates in templates_by_code.items():
        for t in templates:
            idx.add(t, code)
    return idx
