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

# Design tokens. Dark by choice: this is projected in a dim room, and the three
# gate states need to read as colour at a glance from the back of a hall.
# Streamlit's own theme comes from .streamlit/config.toml; this styles the parts
# Streamlit does not own.
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&display=swap');

:root{
  --void:#070B1C; --panel:#0D1229; --line:#1E2749; --panel-2:#141B38;
  --ink:#E8ECFF; --muted:#8A93B8;
  --violet:#7C5CFF; --danger:#FF4D6A; --warn:#FFB020; --ok:#2DD4A7;
}
.stApp{background:var(--void);color:var(--ink)}
.block-container{padding-top:2.2rem;max-width:1180px}

/* ---------- serif display voice, used only for the verdict ---------- */
.serif{font-family:'Fraunces',Charter,'Iowan Old Style',Georgia,serif}

/* ---------- transaction under test ---------- */
.scenario{
  background:var(--panel);border:1px solid var(--line);
  border-left:3px solid var(--violet);
  padding:1.15rem 1.3rem;border-radius:3px;margin:.6rem 0;
}
.scenario .meta{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.72rem;letter-spacing:.1em;color:var(--muted);margin-bottom:.5rem;
}
.scenario .msg{font-size:1.08rem;line-height:1.55;color:var(--ink);margin:.55rem 0}
.scenario .payee{font-size:.85rem;color:var(--muted)}

/* ---------- THE verdict band: the one loud thing on the page ---------- */
.verdict{
  border-radius:3px;padding:1.6rem 1.8rem;margin:1rem 0;
  position:relative;overflow:hidden;
  border:1px solid var(--line);
}
.verdict::before{content:'';position:absolute;inset:0;opacity:.14}
.verdict .kicker{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;
  position:relative;
}
.verdict .headline{
  font-family:'Fraunces',Charter,Georgia,serif;
  font-size:2.9rem;line-height:1.05;margin:.45rem 0 .1rem;
  position:relative;letter-spacing:-.02em;
}
.verdict .detail{font-size:.95rem;position:relative;opacity:.92}

.v-block::before{background:var(--danger)}   .v-block .headline{color:var(--danger)}
.v-stepup::before{background:var(--warn)}    .v-stepup .headline{color:var(--warn)}
.v-allow::before{background:var(--ok)}       .v-allow .headline{color:var(--ok)}

/* risk number, set large in the band's own right column */
.riskbox{text-align:right;position:relative}
.riskbox .num{
  font-family:'Fraunces',Charter,Georgia,serif;font-size:3.4rem;
  line-height:1;letter-spacing:-.03em;
}
.riskbox .cap{font-size:.68rem;letter-spacing:.14em;color:var(--muted);text-transform:uppercase}

/* ---------- evidence rows ---------- */
.sectionlabel{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.7rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted);margin:1.4rem 0 .5rem;
}
.reason{
  background:var(--panel-2);border:1px solid var(--line);
  border-left:2px solid var(--danger);
  padding:.65rem .9rem;border-radius:2px;margin:.3rem 0;
  font-size:.9rem;line-height:1.5;color:var(--ink);
}
.reason .src{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.66rem;color:var(--muted);letter-spacing:.08em;
}
.safe{background:var(--panel-2);border:1px solid var(--line);
      border-left:2px solid var(--ok);padding:.65rem .9rem;
      border-radius:2px;margin:.3rem 0;font-size:.9rem;line-height:1.5}

/* ---------- layer meters ---------- */
.layer{display:flex;align-items:center;gap:.6rem;margin:.4rem 0}
.layer .nm{font-size:.78rem;letter-spacing:.04em;color:var(--muted);
           width:6.4rem;text-transform:uppercase;
           font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.layer .track{flex:1;height:5px;background:var(--panel-2);
              border-radius:2px;overflow:hidden}
.layer .fill{height:100%;background:var(--violet);border-radius:2px}
.layer .vl{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
           font-size:.76rem;color:var(--ink);width:2.4rem;text-align:right}

/* ---------- ring / alert rows ---------- */
.ring{
  background:var(--panel);border:1px solid var(--line);border-radius:3px;
  padding:.85rem 1rem;margin:.4rem 0;display:flex;justify-content:space-between;
  align-items:center;
}
.ring .size{font-family:'Fraunces',Charter,Georgia,serif;font-size:1.6rem;color:var(--ink)}
.ring .desc{font-size:.82rem;color:var(--muted);line-height:1.45}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"]{background:#05081A;border-right:1px solid var(--line)}
section[data-testid="stSidebar"] h1{
  font-family:'Fraunces',Charter,Georgia,serif;font-size:1.55rem;
  font-weight:600;letter-spacing:-.02em;color:var(--ink);
}
section[data-testid="stSidebar"] p{color:var(--muted);font-size:.8rem;line-height:1.55}

/* ---------- widgets ---------- */
.stButton>button{
  background:var(--violet);color:#fff;border:none;border-radius:3px;
  padding:.55rem 1.5rem;font-weight:600;font-size:.92rem;
  box-shadow:0 0 24px rgba(124,92,255,.28);
}
.stButton>button:hover{background:#8F73FF;color:#fff;
                       box-shadow:0 0 30px rgba(124,92,255,.45)}
.stButton>button:focus{outline:2px solid var(--violet);outline-offset:2px}
[data-testid="stMetricValue"]{font-family:'Fraunces',Charter,Georgia,serif;
                               font-size:1.9rem;color:var(--ink)}
[data-testid="stMetricLabel"]{font-size:.72rem;letter-spacing:.1em;
                              text-transform:uppercase;color:var(--muted)}
[data-testid="stMetricDelta"]{font-size:.8rem}
h1,h2,h3{font-family:'Fraunces',Charter,Georgia,serif;font-weight:600;
         letter-spacing:-.02em;color:var(--ink)}
label{color:var(--muted);font-size:.82rem}
hr{border-color:var(--line)}

@media (prefers-reduced-motion:no-preference){
  .verdict{animation:rise .28s ease-out}
  @keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
}
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

    head, vclass, kicker = {
        "BLOCK": ("Payment stopped", "v-block", "blocked at the gate"),
        "STEP_UP": ("Extra check needed", "v-stepup", "held for verification"),
        "ALLOW": ("Payment cleared", "v-allow", "passed the gate"),
    }[action]

    detail = {
        "BLOCK": "Money did not move. Here is the evidence, and what to do instead.",
        "STEP_UP": "Money will move, but only after one more check on the official app.",
        "ALLOW": "No risk signals fired. Known payee, ordinary amount.",
    }[action]

    st.markdown(
        f'<div class="verdict {vclass}">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-end;gap:1.5rem">'
        f'<div>'
        f'<div class="kicker">{kicker}</div>'
        f'<div class="headline">{head}</div>'
        f'<div class="detail">{detail}</div>'
        f'</div>'
        f'<div class="riskbox"><div class="num">{risk_pct:.0f}<span style="font-size:1.4rem">%</span></div>'
        f'<div class="cap">risk score</div></div>'
        f'</div></div>',
        unsafe_allow_html=True)

    safe = {
        "BLOCK": [
            "Do <b>not</b> accept this request — a collect request sends money, it does not receive it.",
            "Open the merchant's official app yourself and check the payment status there.",
            "If you already paid, call your bank now on the number in the app.",
        ],
        "STEP_UP": [
            "Re-authenticate with an OTP on the official app — never read an OTP to anyone.",
            "Verify with the payee on a number you already had, not one in this message.",
            "If it still feels wrong, cancel and check your bank app.",
        ],
        "ALLOW": ["Nothing to do."],
    }[action]

    if verdict.reasons:
        st.markdown('<div class="sectionlabel">What the guard saw</div>', unsafe_allow_html=True)
        for r in verdict.reasons:
            src = r.get("layer", "rule")
            st.markdown(
                f'<div class="reason">{r["text"]}<br><span class="src">{src.upper()} LAYER</span></div>',
                unsafe_allow_html=True)
    else:
        st.markdown('<div class="reason">No risk signals fired. Known payee, ordinary amount.</div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="sectionlabel">What to do instead</div>', unsafe_allow_html=True)
    for s in safe:
        st.markdown(f'<div class="safe">{s}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="reason" style="border-left-color:var(--muted);margin-top:1rem">'
        f'If this is fraud, report it on the National Cyber Crime helpline '
        f'<b>{HELPLINE}</b> — cybercrime.gov.in</div>',
        unsafe_allow_html=True)


def layer_breakdown(verdict) -> None:
    st.markdown('<div class="sectionlabel">How the four layers scored</div>', unsafe_allow_html=True)
    for name, score in verdict.layers.items():
        pct = max(0, min(1.0, score)) * 100
        st.markdown(
            f'<div class="layer"><div class="nm">{name}</div>'
            f'<div class="track"><div class="fill" style="width:{pct:.0f}%"></div></div>'
            f'<div class="vl">{score:.2f}</div></div>',
            unsafe_allow_html=True)
    st.markdown(
        '<div class="reason" style="border-left-color:var(--violet)">'
        'Scores combine in <span class="src">CONFIGS/CONFIG.YAML</span>. Each layer '
        'reports independently, so a decision can always be traced to the evidence '
        'behind it.</div>', unsafe_allow_html=True)


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
        f'<div class="scenario">'
        f'<div class="meta">{t["direction"]} · {t["type_code"]}</div>'
        f'<div class="msg">“{t["message"]}”</div>'
        f'<div class="payee">{t["payee_name"]} · {t["payee_vpa"]} · Rs {t["amount"]:,}</div>'
        f'</div>',
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
        '<div class="reason" style="border-left-color:var(--violet)">'
        'An adversary rewrites the scam script — typos, Hinglish, new wording — '
        '<b>and</b> removes its structural tells: the payee account is aged, the '
        'device is one the user has used before, the amount is modest, and there '
        'is no velocity.</div>', unsafe_allow_html=True)

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

    st.markdown('<div class="sectionlabel">Message before and after the attack</div>',
                unsafe_allow_html=True)
    if m["examples"]:
        e = m["examples"][0]
        st.markdown(f'<div class="reason" style="border-left-color:var(--muted)">'
                    f'<span class="src">ORIGINAL</span><br>{e["before"]}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="reason">'
                    f'<span class="src">ATTACKED</span><br>{e["after"]}</div>',
                    unsafe_allow_html=True)

    st.markdown(
        f'<div class="reason" style="border-left-color:var(--warn)">'
        f'<b>The honest cost.</b> Adapting lifts recall from '
        f'{m["adversarial_recall_before_adaptation"]:.0%} to '
        f'{m["adversarial_recall_after_adaptation"]:.0%}, but false positives move '
        f'{m["legit_fpr_before_adaptation"]:.2%} → {m["legit_fpr_after_adaptation"]:.2%}. '
        'A structural-evasion row is <i>built</i> to look like a genuine payment, so some '
        'of that cost is irreducible for any feature-based system. We report it '
        'rather than tune it away.</div>',
        unsafe_allow_html=True)


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

    st.markdown('<div class="sectionlabel">What the model learned to watch for</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="reason" style="border-left-color:var(--violet)">'
        'The model is linear, so every weight is a readable reason — no '
        'explainability service required.</div>', unsafe_allow_html=True)
    if mpath.exists():
        st.dataframe(
            [{"Feature": f["feature"], "Plain English": FEATURE_REASON.get(f["feature"], ""),
              "Weight": f"{f['weight']:+.3f}"} for f in m["top_features"]],
            use_container_width=True, hide_index=True)

    st.markdown('<div class="sectionlabel">Connected fraud rings</div>', unsafe_allow_html=True)
    rings = ENG.graph.top_rings(5)
    if rings:
        for r in rings:
            st.markdown(
                f'<div class="ring"><div class="size">{r["size"]}</div>'
                f'<div class="desc">accounts acting as one operation · '
                f'<b style="color:var(--danger)">{r["flagged"]}</b> already flagged<br>'
                f'Individually clean. Suspicious together.</div></div>',
                unsafe_allow_html=True)
        st.markdown(
            '<div class="reason" style="border-left-color:var(--muted)">'
            'Each account in a ring shares a device fingerprint, phone, or IP with '
            'accounts the rule layer already blocked. No single transaction here looks '
            'unusual — the pattern is the evidence.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="reason">No rings above the minimum size in this dataset.</div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="sectionlabel">Transaction explorer</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="sectionlabel">Review a single alert</div>', unsafe_allow_html=True)
    flagged = [i for i, a in enumerate(actions) if a != "ALLOW"]
    if flagged:
        pick = st.selectbox("Pick a flagged transaction",
                            flagged, format_func=lambda i: f"{subset[i]['txn_id']} — Rs {subset[i]['amount']:,}")
        t = subset[pick]
        v = ENG.score(t)
        card(v.action, t, v)
        layer_breakdown(v)
    else:
        st.markdown('<div class="reason">Nothing flagged in this window.</div>',
                    unsafe_allow_html=True)


# ------------------------------------------------------------------ nav
PAGES = {
    "Payment Simulation": screen_payment,
    "Adaptivity": screen_adaptivity,
    "Analyst Dashboard": screen_dashboard,
}

st.sidebar.markdown(
    '<div style="font-family:Fraunces,Charter,Georgia,serif;font-size:1.5rem;'
    'font-weight:600;letter-spacing:-.02em;color:#E8ECFF;margin-bottom:.2rem">🛡️ UPIGuard</div>',
    unsafe_allow_html=True)
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