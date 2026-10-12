"""UPIGuard demo — payment simulation, warning screen, analyst dashboard.

Run:  streamlit run app/main.py

Three screens, one file, because the demo is about the *flow* a payment takes,
not about a UI framework. Screen 1 is the one a judge watches; screen 3 is the
one they interrogate when they ask "why did it block that?".

The whole risk engine is numpy + stdlib, so only Streamlit itself needs
installing. If it is missing, every number below still computes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "detector"))
sys.path.insert(0, str(ROOT / "src" / "generator"))

from engine import build_engine_from_data, load_config  # noqa: E402
from features import FEATURE_REASON  # noqa: E402
from mutate import build_adversarial_set, risks_for  # noqa: E402
from taxonomy import SCAM_TYPES  # noqa: E402

st.set_page_config(page_title="UPIGuard", page_icon="🛡️", layout="wide")

# Force light mode. The demo is projected on a projector in front of judges,
# and Streamlit's dark theme puts light-background cards (the alert reasons, the
# scenario card) at unreadable contrast. Explicit colours on every card below
# are the real fix; this just stops the base theme fighting them.
CSS = """
<style>
.stApp{background:#ffffff;color:#0f172a}
.blocked{background:#7f1d1d;color:#ffffff;padding:1.1rem;border-radius:10px;margin:.4rem 0}
.stepup{background:#92400e;color:#ffffff;padding:1.1rem;border-radius:10px;margin:.4rem 0}
.allowed{background:#14532d;color:#ffffff;padding:1.1rem;border-radius:10px;margin:.4rem 0}
/* explicit colours on every element: Streamlit dark mode overrides inherited
   text colour, so the body rule alone is not enough */
.reason{background:#f1f5f9;color:#0f172a;border-left:4px solid #dc2626;
        padding:.55rem .85rem;margin:.3rem 0;border-radius:4px;font-size:.92rem}
.reason b,.reason i{color:#0f172a}
.scenario{background:#eff6ff;color:#0f172a;border-left:4px solid #2563eb;
          padding:.75rem .95rem;border-radius:6px;margin:.5rem 0;font-size:.95rem}
.scenario b,.scenario i,.scenario span{color:#0f172a}
.big{font-size:1.5rem;font-weight:700}
.small{font-size:.8rem;color:#64748b}
h1,h2,h3,p,label{color:#0f172a}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ data load
@st.cache_data(show_spinner="Loading model and dataset...")
def load_all():
    cfg = load_config()
    raw = ROOT / "data" / "raw" / "transactions.jsonl"
    if not raw.exists():
        st.error("Run `python3 src/generator/generate.py` first.")
        st.stop()
    rows = [json.loads(l) for l in raw.open(encoding="utf-8")]
    fraud_templates = {s.code: list(s.message_templates) for s in SCAM_TYPES}
    eng = build_engine_from_data(cfg, rows, fraud_templates)
    return cfg, rows, eng, fraud_templates


cfg, ROWS, ENG, TEMPLATES = load_all()
HELPLINE = cfg.get("demo", {}).get("helpline", "1930")
HELPLINE_URL = "https:// cybercrime.gov.in"


def card(action: str, txn: dict, verdict) -> None:
    """The pre-payment gate UI. This is the demo's hero moment."""
    risk_pct = verdict.risk * 100
    amount = txn["amount"]
    if action == "BLOCK":
        st.markdown(
            f'<div class="blocked"><div class="big">BLOCKED — Rs {amount:,}</div>'
            f'<div>This payment was <b>not</b> completed. '
            f'Risk score {risk_pct:.0f}%.</div></div>', unsafe_allow_html=True)
        safe = [
            "Do <b>not</b> accept this request — a collect request sends money, it does not receive it.",
            "Open the merchant's official app yourself and check the payment status there.",
            "If you already paid, contact your bank immediately on the official number.",
        ]
    elif action == "STEP_UP":
        st.markdown(
            f'<div class="stepup"><div class="big">EXTRA CHECK REQUIRED — Rs {amount:,}</div>'
            f'<div>Risk score {risk_pct:.0f}%. One more verification before this goes through.</div></div>',
            unsafe_allow_html=True)
        safe = [
            "Re-authenticate with an OTP on the official app — do not read an OTP to anyone.",
            "Verify with the payee on a number you already had, not one in this message.",
            "If this feels wrong, cancel and check your bank app.",
        ]
    else:
        st.markdown(
            f'<div class="allowed"><div class="big">PAYMENT ALLOWED — Rs {amount:,}</div>'
            f'<div>Risk score {risk_pct:.0f}%. Looks like normal activity.</div></div>',
            unsafe_allow_html=True)
        safe = ["No action needed."]

    st.markdown("**Why UPIGuard decided this**")
    for r in verdict.reasons:
        st.markdown(f'<div class="reason">{r["text"]}</div>', unsafe_allow_html=True)
    if not verdict.reasons:
        st.markdown('<div class="reason">No risk signals fired. Known payee, normal amount.</div>',
                    unsafe_allow_html=True)

    st.markdown("**What to do instead**")
    for s in safe:
        st.markdown(f"- {s}")

    st.markdown(
        f'<div class="small">If this is fraud, report it on the National Cyber Crime '
        f'helpline <b>{HELPLINE}</b> — cybercrime.gov.in. Bank fraud helplines are '
        f'linked from your bank app.</div>',
        unsafe_allow_html=True)


def layer_breakdown(verdict) -> None:
    st.markdown("**Layer scores**")
    cols = st.columns(4)
    for col, (name, score) in zip(cols, verdict.layers.items()):
        col.metric(name.title(), f"{score:.2f}")
    st.caption("Each layer scores independently; the gate fuses them. "
               "Weights live in configs/config.yaml.")


# ------------------------------------------------------------------ screens
def screen_payment() -> None:
    st.header("① Payment Simulation")
    st.caption("A transaction arrives. The guard scores it before money moves.")

    by_type: dict[str, list[dict]] = {}
    for r in ROWS:
        by_type.setdefault(r["type_code"], []).append(r)

    label = {s.code: s.label for s in SCAM_TYPES}
    label.update({r["type_code"]: r["type_label"] for r in ROWS if r["label"] == "legit"})

    choice = st.selectbox(
        "Pick a scenario",
        sorted(by_type),
        format_func=lambda c: f"{c} — {label.get(c, c)}",
    )

    txn = st.select_slider(
        "Transaction", options=range(len(by_type[choice])),
        format_func=lambda i: f"#{i + 1} of {len(by_type[choice])}",
        value=0,
    )
    t = by_type[choice][txn]

    st.markdown(
        f'<div class="scenario"><b>{t["direction"]}</b> · <b>Rs {t["amount"]:,}</b><br>'
        f'<i>"{t["message"]}"</i><br>'
        f'<span class="small">payee: {t["payee_name"]} ({t["payee_vpa"]})</span></div>',
        unsafe_allow_html=True)

    if st.button("Run the guard", type="primary"):
        v = ENG.score(t)
        st.divider()
        card(v.action, t, v)
        layer_breakdown(v)


def screen_adaptivity() -> None:
    st.header("② Adaptivity — attack, detect, learn")
    st.caption("The number that matters. In-distribution accuracy is flattering; "
               "this is what happens when someone actually attacks the system.")

    p = ROOT / "data" / "processed" / "adaptivity.json"
    if not p.exists():
        st.warning("Run `python3 src/detector/mutate.py` first.")
        st.stop()
    m = json.loads(p.read_text())

    st.markdown(
        '<div class="scenario">An adversary rewrites the scam script (typos, '
        'Hinglish, new wording) <b>and</b> removes its structural tells: the payee '
        'account is aged, the device is one the user has used before, the amount is '
        'modest, and there is no velocity.</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    cols[0].metric("Clean recall", f"{m['clean_fraud_recall']:.0%}")
    cols[1].metric("Under attack", f"{m['adversarial_recall_before_adaptation']:.0%}",
                   delta=f"{m['adversarial_recall_before_adaptation'] - m['clean_fraud_recall']:+.0%}",
                   delta_color="inverse")
    cols[2].metric("After analyst review", f"{m['adversarial_recall_after_adaptation']:.0%}",
                   delta=f"+{m['recovery_percentage_points']:.1f} pp")
    cols[3].metric("Misses reviewed", m["misses_before_adaptation"])

    st.markdown("**Recall per scam family, under attack**")
    rows = [{"Code": c, "Family": SCAM_BY.get(c, c), "Caught": d["caught"],
             "Total": d["n"], "Recall": f"{d['recall']:.0%}"}
            for c, d in m["per_type_adversarial"].items()]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("**Example of an attacked message**")
    if m["examples"]:
        e = m["examples"][0]
        st.markdown(f'<div class="reason"><b>Original:</b> {e["before"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="reason"><b>Attacked:</b> {e["after"]}</div>', unsafe_allow_html=True)

    st.info(
        f"**The honest cost.** Adapting lifts recall from "
        f"{m['adversarial_recall_before_adaptation']:.0%} to "
        f"{m['adversarial_recall_after_adaptation']:.0%}, but false positives move "
        f"{m['legit_fpr_before_adaptation']:.2%} → {m['legit_fpr_after_adaptation']:.2%}. "
        "A structural-evasion row is *built* to look like a genuine payment, so some "
        "of that cost is irreducible for any feature-based system. We report it "
        "rather than tune it away.")


SCAM_BY = {s.code: s.label for s in SCAM_TYPES}


def screen_dashboard() -> None:
    st.header("③ Analyst Dashboard")
    st.caption("Every alert, drillable. Click through to the evidence.")

    mpath = ROOT / "data" / "processed" / "metrics.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        c = st.columns(4)
        c[0].metric("Transactions", f"{m['n_transactions']:,}")
        c[1].metric("Fraud recall", f"{m['overall']['recall']:.1%}")
        c[2].metric("False positive rate", f"{m['overall']['false_positive_rate']:.2%}",
                    help="Share of genuine payments the system would stop. This is "
                         "the cost of being wrong.")
        c[3].metric("Latency / txn", f"{m['latency_ms_per_txn']:.2f} ms")

    st.markdown("**Top learned features** (a linear model, so every weight is readable)")
    if mpath.exists():
        st.dataframe(
            [{"Feature": f["feature"], "Plain English": FEATURE_REASON.get(f["feature"], ""),
              "Weight": f"{f['weight']:+.3f}"} for f in m["top_features"]],
            use_container_width=True, hide_index=True)

    st.markdown("**Connected fraud rings**")
    rings = ENG.graph.top_rings(5)
    if rings:
        for r in rings:
            st.markdown(
                f'<div class="reason">Ring of <b>{r["size"]}</b> accounts, '
                f'<b>{r["flagged"]}</b> already flagged. Individually clean; '
                f'together they move money as one operation.</div>', unsafe_allow_html=True)
        st.caption("Each account in a ring shares a device fingerprint, phone, or IP "
                   "with accounts the rule layer has already blocked. No single "
                   "transaction here looks unusual — the pattern is the evidence.")
    else:
        st.info("No rings above the minimum size in this dataset.")

    st.markdown("**Transaction explorer**")
    n_show = st.slider("Rows to scan", 50, 800, 200, 50)
    subset = ROWS[:n_show]
    risks = np.array([ENG.score(t).risk for t in subset])
    labels = [("fraud" if t["label"] == "fraud" else "legit") for t in subset]
    actions = ["BLOCK" if r >= cfg["gate"]["block_threshold"] else
               "STEP_UP" if r >= cfg["gate"]["stepup_threshold"] else "ALLOW" for r in risks]
    df = [{"Txn": t["txn_id"], "Truth": lab, "Action": act, "Risk": f"{r:.2f}",
           "Amount": f"Rs {t['amount']:,}", "Type": t["type_label"],
           "Message": t["message"][:70]}
          for t, lab, act, r in zip(subset, labels, actions, risks)]
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("**Review a single alert**")
    flagged = [i for i, a in enumerate(actions) if a != "ALLOW"]
    if flagged:
        pick = st.selectbox("Pick a flagged transaction",
                            flagged, format_func=lambda i: f"{subset[i]['txn_id']} — Rs {subset[i]['amount']:,}")
        t = subset[pick]
        v = ENG.score(t)
        card(v.action, t, v)
        layer_breakdown(v)
    else:
        st.info("Nothing flagged in this window.")


# ------------------------------------------------------------------ nav
PAGES = {
    "Payment Simulation": screen_payment,
    "Adaptivity": screen_adaptivity,
    "Analyst Dashboard": screen_dashboard,
}

st.sidebar.title("🛡️ UPIGuard")
st.sidebar.caption("Pre-payment fraud guard for UPI · CyberCup 6.0 · CC6-AI02")
choice = st.sidebar.radio("Screen", list(PAGES))
st.sidebar.divider()
st.sidebar.markdown(
    "**Prototype scope**\n\n"
    "**Can:** score a simulated UPI transaction, block/step-up/allow with "
    "readable reasons, detect 10 fraud families and linked mule rings, adapt "
    "to newly-labelled scams.\n\n"
    "**Cannot:** run inside a real UPI app, see the live payment rail, verify "
    "bank KYC, or catch an attack where the payer knowingly initiates the "
    "transfer to a genuinely new payee.")

PAGES[choice]()