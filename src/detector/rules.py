"""Deterministic rule engine.

High precision, every hit carries a human-readable reason. Rules live in
config.yaml so the analyst feedback loop can edit them without a code change —
that edit is part of the adaptivity demo.

Each rule returns (name, score) where score is its confidence that this
transaction is fraud *given the rule matched*. The engine does not sum them
(that's what the fusion layer is for); it reports the max as its layer score
and keeps every match as a reason.
"""

from __future__ import annotations

import re

CRED_RE = re.compile(r"\b(otp|upi pin|cvv|password|6-digit)\b")
REFUND_WORDS = ("refund", "credit", "credited", "reversed", "in your favour", "won")
URGENCY_WORDS = ("urgent", "immediately", "within 24 hours", "right now", "10 minutes",
                 "30 minutes", "limited", "avoid a penalty", "suspended", "expire")
BRANDS = ("hdfc", "icici", "sbi bank", "axis bank", "kotak", "rbi", "paytm", "phonepe")


def evaluate(txn: dict, cfg: dict) -> tuple[float, list[dict]]:
    """Return (layer_score, reasons) for one transaction.

    layer_score = strongest single rule confidence (0.0 if nothing matched).
    Using max rather than sum stops a long message from accumulating enough
    weak keyword hits to look fraudulent on its own.
    """
    rules = cfg.get("rules", {})
    text = (txn.get("message") or "").lower()
    direction = txn.get("direction", "PAY")
    amount = float(txn.get("amount", 0) or 0)
    reasons: list[dict] = []

    def hit(name: str, detail: str) -> None:
        reasons.append({"rule": name, "score": rules.get(name, 0.5), "detail": detail})

    # --- the headline catch: money-moving request phrased as money-arriving
    is_collect = direction == "COLLECT_REQUEST"
    claims_incoming = any(w in text for w in REFUND_WORDS)
    if is_collect and claims_incoming:
        hit("collect_request_refund",
            "Message promises a refund/credit, but a collect request sends money OUT.")

    # --- asks for a secret alongside or instead of a payment
    cred_hits = CRED_RE.findall(text)
    if len(cred_hits) >= 2 or (cred_hits and is_collect):
        hit("credential_request",
            f"Requests credentials ({', '.join(sorted(set(cred_hits)))}). No legitimate flow needs these.")

    # --- QR payload disagrees with the merchant the user was told about
    claimed = (txn.get("qr_claimed_merchant") or "").strip().lower()
    payload = (txn.get("qr_payload_merchant") or "").strip().lower()
    if txn.get("qr_present") and claimed and payload and claimed != payload:
        hit("qr_merchant_mismatch",
            f"QR says '{payload}' but the merchant shown was '{claimed}'.")

    # --- brand spoofing: too many official-sounding names in one short message
    brand_hits = [b for b in BRANDS if b in text]
    if len(brand_hits) >= 2:
        hit("brand_impersonation",
            f"References {len(brand_hits)} banks/bodies ({', '.join(brand_hits)}).")

    # --- urgency pressure
    urg_hits = [w for w in URGENCY_WORDS if w in text]
    if len(urg_hits) >= 2:
        hit("urgency_keywords", f"Urgency pressure: {', '.join(urg_hits[:3])}.")

    # --- behavioural-ish rules that stay deterministic
    prior = float(txn.get("prior_txn_count_with_payee", 0) or 0)
    payee_age = float(txn.get("payee_account_age_days", 365) or 0)
    if prior == 0 and amount >= 5000:
        hit("new_payee_first_large",
            f"First-ever payment, large amount (Rs {amount:,.0f}), payee account {payee_age:.0f}d old.")
    if prior == 0 and payee_age < 30 and amount >= 1000:
        hit("new_payee_first_large",
            f"New payee ({payee_age:.0f}d old account) receiving Rs {amount:,.0f}.")

    if txn.get("new_device_flag") and amount >= 3000:
        hit("new_device", "Large payment from a device not seen before on this account.")
    if txn.get("emulator_flag"):
        hit("new_device", "Device fingerprint looks like an emulator.")

    if float(txn.get("txn_count_last_hour", 0) or 0) >= 15:
        hit("high_velocity", f"{txn.get('txn_count_last_hour')} transactions in the last hour.")

    if amount >= 3000 and amount % 100 == 0:
        hit("round_amount_high", f"Round amount of Rs {amount:,.0f}.")

    if not reasons:
        return 0.0, []

    top = max(r["score"] for r in reasons)
    return float(top), reasons
