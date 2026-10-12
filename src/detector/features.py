"""Feature extraction for the behavioural model.

Design constraint: every feature is *named after what it means*, so the fitted
linear model can report its top contributions as reason codes a user can read.
That is why this is a linear model over hand-built features rather than a GBM
over anonymous columns — a judge reading the alert needs "first payment to a
6-day-old payee", not "feature_17 = 0.8".

Stdlib + numpy only, on purpose: the risk engine must run with no ML stack.
"""

from __future__ import annotations

import math

import numpy as np

# Ordered. Index in this list == column index in the design matrix, so model
# weights stay interpretable and the reason-code path needs no lookup table.
FEATURE_NAMES: list[str] = [
    "collect_request",              # money leaves the account
    "incoming_direction",
    "purpose_direction_mismatch",   # "refund" wording but a collect request
    "amount_log",                   # log-scaled, keeps one huge txn from dominating
    "round_amount",
    "new_payee",                    # never paid this VPA before
    "new_payee_large",              # new payee AND first big amount
    "payee_account_age_log",        # log-age; fraud payees are days old, not years
    "credential_request",           # OTP / PIN / CVV mentioned
    "urgency_keywords",
    "qr_merchant_mismatch",
    "qr_present",
    "new_device",
    "emulator",
    "txn_per_10min",                # velocity
    "txn_per_hour",
    "distinct_payees_recent",
    "brand_impersonation",
]

# Weights a term must carry for the text to count as a credential request. Kept
# explicit so the rule can be argued about in the report rather than hidden in
# a regex.
CREDENTIAL_TERMS = (
    "otp", "upi pin", "pin", "cvv", "password", "share the otp",
    "6-digit", "verify your identity", "authorise",
)

URGENCY_TERMS = (
    "within 24 hours", "immediately", "urgent", "right now", "limited time",
    "do it now", "today itself", "10 minutes", "30 minutes", "before the due date",
    "avoid a penalty", "will be suspended", "pending complaint", "expire",
)

BRAND_TERMS = (
    "hdfc", "icici", "sbi", "axis", "kotak", "bank", "rbi", "upi", "kyc",
    "aadhaar", "amazon", "flipkart", "zomato", "irctc", "dmart", "metro mart",
)


def _text(txn: dict) -> str:
    return (txn.get("message") or "").lower()


def _term_count(text: str, terms: tuple[str, ...]) -> int:
    return sum(1 for t in terms if t in text)


def extract(txn: dict, user_stats: dict | None = None) -> np.ndarray:
    """Turn one transaction into the feature vector used by the linear model.

    `user_stats` carries rolling counts the caller maintains (recent distinct
    payees, user mean amount). It is optional so the demo can score a single
    transaction without a history.
    """
    stats = user_stats or {}
    text = _text(txn)
    direction = txn.get("direction", "PAY")
    amount = float(txn.get("amount", 0) or 0)

    is_collect = 1.0 if direction == "COLLECT_REQUEST" else 0.0
    is_incoming = 1.0 if direction == "INCOMING" else 0.0

    # The core mismatch: refund/credit wording that would make a user expect
    # money to arrive, attached to a request that moves money out.
    claims_incoming = any(w in text for w in ("refund", "credit", "credited", "reversed", "won"))
    mismatch = 1.0 if (is_collect and claims_incoming) else 0.0

    prior = float(txn.get("prior_txn_count_with_payee", 0) or 0)
    new_payee = 1.0 if prior == 0 else 0.0

    user_mean = float(stats.get("mean_amount", 0) or 0)
    if user_mean > 0:
        large = 1.0 if amount > 3 * user_mean else 0.0
    else:
        large = 1.0 if amount >= 5000 else 0.0

    payee_age = float(txn.get("payee_account_age_days", 365) or 0)
    age_log = math.log1p(max(0.0, payee_age))

    qr_claimed = (txn.get("qr_claimed_merchant") or "").strip().lower()
    qr_payload = (txn.get("qr_payload_merchant") or "").strip().lower()
    qr_present = 1.0 if txn.get("qr_present") else 0.0
    # Only a *difference* counts; a present QR alone is normal commerce.
    mismatch_qr = 1.0 if (qr_present and qr_claimed and qr_payload and qr_claimed != qr_payload) else 0.0

    cred = 1.0 if _term_count(text, CREDENTIAL_TERMS) >= 2 else 0.0
    urg = min(1.0, _term_count(text, URGENCY_TERMS) / 2.0)
    brand = 1.0 if _term_count(text, BRAND_TERMS) >= 3 else 0.0

    return np.array([
        is_collect,
        is_incoming,
        mismatch,
        math.log1p(amount),
        1.0 if amount % 100 == 0 else 0.0,
        new_payee,
        new_payee * large,
        age_log,
        cred,
        urg,
        mismatch_qr,
        qr_present,
        1.0 if txn.get("new_device_flag") else 0.0,
        1.0 if txn.get("emulator_flag") else 0.0,
        float(txn.get("txn_count_last_10min", 0) or 0) / 5.0,
        float(txn.get("txn_count_last_hour", 0) or 0) / 30.0,
        float(stats.get("distinct_payees_recent", 0) or 0) / 5.0,
        brand,
    ], dtype=np.float64)


def extract_many(txns: list[dict], user_stats: dict | None = None) -> np.ndarray:
    if not txns:
        return np.zeros((0, len(FEATURE_NAMES)))
    return np.vstack([extract(t, user_stats) for t in txns])


def user_amount_stats(rows: list[dict], user_key: str = "user_home_vpa") -> dict:
    """Mean amount per user, used for the "unusually large for you" feature."""
    totals: dict[str, list[float]] = {}
    for r in rows:
        totals.setdefault(r.get(user_key, "anon"), []).append(float(r.get("amount", 0) or 0))
    return {u: {"mean_amount": float(np.mean(v))} for u, v in totals.items()}


# Human-readable phrasing for each reason code emitted by the model.
FEATURE_REASON: dict[str, str] = {
    "collect_request": "This is a collect request, so money will leave your account",
    "incoming_direction": "Incoming transaction",
    "purpose_direction_mismatch": "Message says refund/credit but the request sends money",
    "amount_log": "Amount magnitude",
    "round_amount": "Round, notable amount",
    "new_payee": "You have never paid this account before",
    "new_payee_large": "First payment to this new payee is large",
    "payee_account_age_log": "Payee account is new",
    "credential_request": "Asks for OTP / PIN / credentials",
    "urgency_keywords": "Creates urgency or a deadline",
    "qr_merchant_mismatch": "QR payload does not match the claimed merchant",
    "qr_present": "Paid via QR code",
    "new_device": "Payment made from a new device",
    "emulator": "Device looks like an emulator",
    "txn_per_10min": "Many payments in a short window",
    "txn_per_hour": "High payment volume today",
    "distinct_payees_recent": "Many different payees in a short window",
    "brand_impersonation": "Mentions several brands or official-sounding terms",
}
