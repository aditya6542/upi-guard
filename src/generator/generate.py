"""Synthetic transaction generator.

Emits a labelled JSONL dataset of UPI transactions. Two properties matter for the
project and are not automatic:

1. **Fraud rings.** Fraud does not arrive as isolated transactions. A ring shares
   one device fingerprint, one phone, and one QR merchant across many accounts, so
   that individually-clean accounts become suspicious only when viewed together.
   The graph detector needs this to have something real to find.

2. **Realistic class balance.** Fractions of a percent fraud is realistic but makes
   per-class metrics unstable, so we oversample to a moderate ratio and report the
   false-positive rate at the chosen operating point.

The same generator drives the Streamlit demo, so the demo data and the test set
never drift apart.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from taxonomy import LEGIT_TYPES, SCAM_BY_CODE, SCAM_TYPES, LegitType, ScamType

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ identities
BRANDS = ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK", "PNB", "BOI", "CANARA"]

MERCHANTS_LEGIT = [
    "Metro Mart", "Reliance Fresh", "Cafe Blue", "BigBasket", "DMart", "Amazon",
    "Flipkart", "IRCTC", "MTR", "Zomato", "Swiggy", "Airtel Recharge",
    "BESCOM", "Apollo Pharmacy", "Starbucks", "PVR Cinemas",
]

# Fraud merchants mimic real brands — this is what makes QR-swap hard for a user
# to spot by eye, and what makes the payload-vs-claimed-name mismatch the tell.
MERCHANTS_FRAUD = [
    ("METRO MART", "Metro Mart"), ("DMART", "DMart"), ("Amazon Pay", "Amazon"),
    ("Zomato Food", "Zomato"), ("IRCTC Rail", "IRCTC"), ("HDFC Bank", "HDFC Bank"),
    ("KYC Update Cell", "KYC Update Cell"), ("Prize Desk India", "Prize Desk India"),
    ("Easy Task Jobs", "Easy Task Jobs"),
]


@dataclass
class FraudRing:
    """A group of mule accounts operated by one fraudster."""

    ring_id: str
    device_id: str
    ip_address: str
    merchant_pair: tuple[str, str]
    payees: list[str] = field(default_factory=list)
    members: list[str] = field(default_factory=list)


def _phone(rng: random.Random) -> str:
    return f"9{rng.randint(10, 99)}{rng.randint(10000000, 99999999)}"


def _vpa(rng: random.Random, name: str) -> str:
    handle = name.lower().replace(" ", "").replace("&", "")[:18]
    return f"{handle}{rng.randint(10, 99)}@okhdfcbank"


def _pick_amount(rng: random.Random, lo: int, hi: int, round_bias: float) -> int:
    """Bounded by type, but biased toward round amounts — fraud amounts are chosen
    to look notable (4999, 12000), genuine ones span a realistic distribution."""
    if rng.random() < round_bias:
        magnitude = 10 ** rng.randint(2, 5)
        candidates = [m * i for m in (magnitude,) for i in range(1, 10)]
        candidates = [c for c in candidates if lo <= c <= hi]
        if candidates:
            return rng.choice(candidates)
    return rng.randint(lo, hi)


def _age_days(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(lo, hi)


def _base_txn_id(index: int) -> str:
    return f"TXN{index:07d}"


# ------------------------------------------------------------------ generation
def generate(n_transactions: int, fraud_ratio: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    idx = 0
    now = datetime(2026, 10, 1, 9, 0, 0)

    n_fraud = int(n_transactions * fraud_ratio)
    n_legit = n_transactions - n_fraud

    # ------------------------------------------------ legit transactions
    # Legit users get stable device/phone identities so that the graph detector
    # sees them as isolated nodes. If legit users shared devices the graph layer
    # would fire constantly and false positives would be meaningless.
    legit_users: list[dict] = []
    for i in range(max(1, n_legit // 8)):
        legit_users.append({
            "device_id": f"DEV-LEG-{i:04d}",
            "phone": _phone(rng),
            "ip": f"49.36.{rng.randint(0,255)}.{rng.randint(1,254)}",
            "home_vpa": _vpa(rng, rng.choice(["rahul", "priya", "amit", "sneha", "karan"])),
        })

    for _ in range(n_legit):
        idx += 1
        utype: LegitType = rng.choice(LEGIT_TYPES)
        user = rng.choice(legit_users)
        amount = _pick_amount(rng, *utype.amount_range, utype.round_amount_bias)
        payee_name = rng.choice(MERCHANTS_LEGIT)
        rows.append({
            "txn_id": _base_txn_id(idx),
            "label": "legit",
            "type_code": utype.code,
            "type_label": utype.label,
            "direction": utype.direction,
            "amount": amount,
            "message": rng.choice(utype.message_templates).format(amount=amount),
            "qr_present": utype.direction == "PAY" and rng.random() < 0.3,
            "qr_claimed_merchant": payee_name if utype.direction == "PAY" else None,
            "qr_payload_merchant": payee_name if utype.direction == "PAY" else None,
            "payee_vpa": _vpa(rng, payee_name),
            "payee_name": payee_name,
            "payee_account_age_days": _age_days(rng, *utype.account_age_days),
            "user_account_age_days": _age_days(rng, 365, 3650),
            "prior_txn_count_with_payee": rng.randint(0, 40),
            "user_device_id": user["device_id"],
            "user_phone": user["phone"],
            "user_ip": user["ip"],
            "user_home_vpa": user["home_vpa"],
            "new_device_flag": False,
            "emulator_flag": False,
            "ring_id": None,
            "txn_count_last_10min": rng.randint(0, 2),
            "txn_count_last_hour": rng.randint(0, 12),
            "created_at": (now - timedelta(minutes=rng.randint(0, 60 * 24 * 30))).isoformat(),
        })

    # ------------------------------------------------ fraud rings
    # Fraud is generated ring-first, not transaction-first. Each ring owns one
    # device, one IP and one merchant QR; its member accounts receive pass-through
    # money and forward it. This is what the graph layer detects.
    rings: list[FraudRing] = []
    n_rings = max(3, n_fraud // 120)
    for r in range(n_rings):
        pair = rng.choice(MERCHANTS_FRAUD)
        rings.append(FraudRing(
            ring_id=f"RING-{r:03d}",
            device_id=f"DEV-FRAUD-{r:04d}",
            ip_address=f"103.21.{rng.randint(0,255)}.{rng.randint(1,254)}",
            merchant_pair=pair,
        ))

    # ring members receive money (mule activity) and forward it
    fraud_rows: list[dict] = []
    for ring in rings:
        n_members = rng.randint(8, 30)
        for m in range(n_members):
            member_vpa = _vpa(rng, f"user{m:03d}")
            ring.members.append(member_vpa)
            ring.payees.append(_vpa(rng, "receiver"))

    # Reserve a share of the fraud budget for single-shot scripts. Without this,
    # the ring transactions can consume the whole budget and the scattered types
    # (lottery, task, refund) never appear in the dataset at all.
    n_singles_budget = int(n_fraud * 0.45)
    n_ring_budget = n_fraud - n_singles_budget

    per_ring = n_ring_budget // max(1, len(rings))
    ring_scam_codes = ("F05", "F04", "F07", "F06", "F03")
    for ring_idx, ring in enumerate(rings):
        # Round-robin rather than random choice: with only ~12 rings a random pick
        # can leave an entire scam type with zero ring examples.
        stype: ScamType = SCAM_BY_CODE[ring_scam_codes[ring_idx % len(ring_scam_codes)]]
        for k in range(per_ring):
            idx += 1
            amount = _pick_amount(rng, *stype.amount_range, stype.round_amount_bias)
            member = rng.choice(ring.members)
            rows.append({
                "txn_id": _base_txn_id(idx),
                "label": "fraud",
                "type_code": stype.code,
                "type_label": stype.label,
                "direction": stype.direction,
                "amount": amount,
                "message": rng.choice(stype.message_templates).format(amount=amount),
                "qr_present": rng.random() < 0.35,
                "qr_claimed_merchant": ring.merchant_pair[0],
                "qr_payload_merchant": ring.merchant_pair[0],
                "payee_vpa": rng.choice(ring.payees),
                "payee_name": ring.merchant_pair[1],
                "payee_account_age_days": _age_days(rng, *stype.account_age_days),
                "user_account_age_days": _age_days(rng, *stype.account_age_days),
                "prior_txn_count_with_payee": rng.randint(0, 2),
                "user_device_id": ring.device_id,
                "user_phone": _phone(rng),
                "user_ip": ring.ip_address,
                "user_home_vpa": member,
                "new_device_flag": rng.random() < 0.6,
                "emulator_flag": rng.random() < 0.15,
                "ring_id": ring.ring_id,
                "txn_count_last_10min": rng.randint(1, 9),
                "txn_count_last_hour": rng.randint(5, 60),
                "created_at": (now - timedelta(minutes=rng.randint(0, 60 * 24 * 30))).isoformat(),
            })
            fraud_rows.append(idx)

    # ------------------------------------------------ scattered single-shot fraud
    # Not all fraud is organised. These are the one-off scripts (lottery, task,
    # QR swap) where the per-transaction layers must carry the detection.
    singles: list[ScamType] = [s for s in SCAM_TYPES if s.code in
                               ("F01", "F02", "F08", "F09", "F10")]
    remaining = n_fraud - len(fraud_rows)
    for k in range(max(0, remaining)):
        idx += 1
        stype = rng.choice(singles)
        amount = _pick_amount(rng, *stype.amount_range, stype.round_amount_bias)
        claimed, payload = rng.choice(MERCHANTS_FRAUD)
        # QR swap: the claimed merchant is real, the payload is the fraudster's.
        qr_mismatch = stype.code == "F02" and rng.random() < 0.7
        rows.append({
            "txn_id": _base_txn_id(idx),
            "label": "fraud",
            "type_code": stype.code,
            "type_label": stype.label,
            "direction": stype.direction,
            "amount": amount,
            "message": rng.choice(stype.message_templates).format(amount=amount),
            "qr_present": rng.random() < 0.5,
            "qr_claimed_merchant": claimed,
            "qr_payload_merchant": (payload + " X") if qr_mismatch else payload,
            "payee_vpa": _vpa(rng, payload),
            "payee_name": payload,
            "payee_account_age_days": _age_days(rng, *stype.account_age_days),
            "user_account_age_days": _age_days(rng, 200, 3650),
            "prior_txn_count_with_payee": rng.randint(0, 1),
            "user_device_id": f"DEV-FRAUD-S{rng.randint(1000,9999)}",
            "user_phone": _phone(rng),
            "user_ip": f"182.7{rng.randint(0,255)}.{rng.randint(1,254)}",
            "user_home_vpa": _vpa(rng, "victim"),
            "new_device_flag": rng.random() < 0.4,
            "emulator_flag": rng.random() < 0.05,
            "ring_id": None,
            "txn_count_last_10min": rng.randint(0, 4),
            "txn_count_last_hour": rng.randint(0, 25),
            "created_at": (now - timedelta(minutes=rng.randint(0, 60 * 24 * 30))).isoformat(),
        })

    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--fraud-ratio", type=float, default=0.18)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(ROOT / "data/raw/transactions.jsonl"))
    args = ap.parse_args()

    rows = generate(args.n, args.fraud_ratio, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_fraud = sum(1 for r in rows if r["label"] == "fraud")
    rings = {r["ring_id"] for r in rows if r.get("ring_id")}
    print(f"wrote {len(rows)} transactions -> {out}")
    print(f"  fraud={n_fraud} ({n_fraud/len(rows):.1%})  legit={len(rows)-n_fraud}")
    print(f"  fraud rings={len(rings)}")
    from collections import Counter
    for code, c in sorted(Counter(r["type_code"] for r in rows).items()):
        print(f"    {code}: {c}")


if __name__ == "__main__":
    main()
