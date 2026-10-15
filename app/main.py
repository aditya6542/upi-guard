"""Bounce'D — Intelligent UPI Fraud Detection (demo).

Entry: streamlit run app/main.py
Backend: GuardEngine.score(txn) -> Verdict(risk, action, layers, reasons).
All numbers below come from the real engine + precomputed metrics files.
"""

from __future__ import annotations

import html
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
from taxonomy import SCAM_TYPES  # noqa: E402

st.set_page_config(page_title="Bounce'D — UPI Fraud Detection", page_icon="🛡️", layout="wide")

# ------------------------------------------------------------- design system
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
.stApp{background:#050505;color:#F5F5F5;font-family:'Inter',system-ui,sans-serif}
header[data-testid="stHeader"]{display:none!important}
div[data-testid="stToolbar"]{display:none!important}
div[data-testid="stDecoration"]{display:none!important}
div[data-testid="stStatusWidget"]{display:none!important}
.block-container{padding-top:2.5rem!important;max-width:1280px;margin-top:0!important}
h1,h2,h3{color:#F5F5F5!important;font-weight:800;letter-spacing:-.02em}
.tech{font-family:'JetBrains Mono',monospace;font-size:.68rem;letter-spacing:.12em;
  text-transform:uppercase;color:#92978E}
.hdr{border:1px solid #252A24;background:#0B0D0B;padding:.9rem 1.2rem;
  display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap}
.logo{font-weight:800;font-size:1.25rem;letter-spacing:-.02em}
.logo b{color:#C6FF00}
.live{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:.65rem;
  letter-spacing:.14em;border:1px solid #C6FF00;color:#C6FF00;border-radius:99px;
  padding:.2rem .7rem;margin-left:.6rem}
.dot{display:inline-block;width:8px;height:8px;border-radius:99px;background:#C6FF00;
  box-shadow:0 0 10px #C6FF00;margin-right:.4rem;animation:pulse 2s infinite}
@keyframes pulse{50%{opacity:.4}}
.panel{background:#0B0D0B;border:1px solid #252A24;padding:1.1rem 1.2rem;margin-bottom:1rem}
.panel h3{margin:0 0 .2rem;font-size:1rem}
.kpi-v{font-size:2rem;font-weight:800;letter-spacing:-.02em}
.kpi-v small{font-size:.9rem;color:#92978E}
.stButton>button{background:#C6FF00!important;color:#0A0C00!important;border:none!important;
  border-radius:6px!important;font-weight:800!important;padding:.65rem 1rem!important;
  width:100%;transition:box-shadow .2s}
.stButton>button:hover{box-shadow:0 0 24px rgba(198,255,0,.45)!important}
.stSelectbox>div>div,.stSlider>div{color:#F5F5F5}
section[data-testid="stSidebar"]{display:none}
/* ---------- iPhone 14/15 frame: 393x852 logical, aspect ~0.461 ---------- */
.iphone{width:min(340px,100%);aspect-ratio:393/852;margin:0 auto;background:#1a1a1c;
  border-radius:54px;padding:11px;box-shadow:0 0 0 2px #3a3a3c,0 0 0 4px #000,
  0 24px 70px rgba(0,0,0,.6),0 0 60px rgba(198,255,0,.06);box-sizing:border-box;
  position:relative}
.iphone::before{content:'';position:absolute;right:-3px;top:170px;width:3px;height:64px;
  background:#3a3a3c;border-radius:0 2px 2px 0}
.iphone::after{content:'';position:absolute;left:-3px;top:140px;width:3px;height:100px;
  background:#3a3a3c;border-radius:2px 0 0 2px}
.screen{background:#0B0D0B;border-radius:44px;overflow:hidden;height:100%;
  display:flex;flex-direction:column;position:relative}
.island{position:absolute;top:12px;left:50%;transform:translateX(-50%);width:100px;
  height:26px;background:#000;border-radius:99px;z-index:5}
.statusbar{display:flex;justify-content:space-between;align-items:center;
  padding:.7rem 1.4rem .1rem;font-size:.68rem;font-weight:600;color:#F5F5F5}
.phead{padding:.5rem 1rem .6rem;border-bottom:1px solid #252A24;text-align:center}
.pamt{font-size:1.7rem;font-weight:800}
.pbody{padding:.7rem .9rem;flex:1;overflow-y:auto}
.homebar{width:110px;height:4px;background:#3a3a3c;border-radius:99px;margin:.4rem auto .5rem}
.riskring{display:flex;justify-content:center;padding:.3rem 0}
.msg{background:#111511;border:1px solid #252A24;border-radius:8px;padding:.6rem .75rem;
  font-size:.78rem;color:#D6D9D2;line-height:1.5}
.kv{display:flex;justify-content:space-between;font-size:.74rem;padding:.35rem 0;
  border-bottom:1px dashed #252A24;color:#92978E}
.kv b{color:#F5F5F5}
.sheet{border-radius:12px 12px 0 0;margin-top:.8rem;padding:.85rem .9rem;
  border-top:3px solid;animation:rise .25s ease-out}
@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.s-block{background:#1A0B0D;border-top-color:#F04438}
.s-step{background:#1A1206;border-top-color:#F79009}
.s-allow{background:#07130D;border-top-color:#12B76A}
.bdg{display:inline-block;font-size:.64rem;font-weight:800;letter-spacing:.1em;
  padding:.2rem .6rem;border-radius:99px;text-transform:uppercase}
.b-block{background:rgba(240,68,56,.15);color:#F97066;border:1px solid #F04438}
.b-step{background:rgba(247,144,9,.15);color:#FDB022;border:1px solid #F79009}
.b-allow{background:rgba(18,183,106,.15);color:#32D583;border:1px solid #12B76A}
.ev{font-size:.72rem;padding:.4rem .55rem;border-radius:6px;margin:.3rem 0;line-height:1.45}
.e-block{background:rgba(240,68,56,.1);border-left:2px solid #F04438;color:#FECDC2}
.e-step{background:rgba(247,144,9,.1);border-left:2px solid #F79009;color:#FEDF89}
.e-allow{background:rgba(18,183,106,.1);border-left:2px solid #12B76A;color:#ABEFC6}
.rbar{height:6px;background:#1E221D;border-radius:99px;overflow:hidden;margin:.4rem 0}
.rbar div{height:100%}
.hl{background:#000;border:1px solid #252A24;border-radius:8px;padding:.5rem;
  font-size:.7rem;text-align:center;color:#92978E;margin-top:.5rem}
.hl b{color:#C6FF00}
.lyr{display:flex;align-items:center;gap:.5rem;margin:.32rem 0;font-size:.72rem}
.lyr .nm{width:7rem;color:#92978E;font-family:'JetBrains Mono',monospace;
  font-size:.64rem;text-transform:uppercase}
.lyr .tr{flex:1;height:5px;background:#1E221D;border-radius:99px;overflow:hidden}
.lyr .fl{height:100%;background:#C6FF00}
.lyr .vl{width:2.6rem;text-align:right;font-family:'JetBrains Mono',monospace}
table{width:100%;font-size:.78rem;border-collapse:collapse}
th{font-family:'JetBrains Mono',monospace;font-size:.62rem;letter-spacing:.1em;
  text-transform:uppercase;color:#92978E;text-align:left;padding:.4rem .5rem;
  border-bottom:1px solid #252A24}
td{padding:.42rem .5rem;border-bottom:1px solid #1A1D19;color:#D6D9D2}
.pos{color:#32D583;font-weight:700}.neg{color:#F97066;font-weight:700}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
/* ---------- overlap / mobile fixes ---------- */
.block-container{overflow-x:hidden}
[data-testid="column"]{min-width:0;overflow:hidden}
.iphone{width:min(340px,100%)}
.screen{min-height:auto;max-height:none;overflow:visible}
.msg,.kv,.ev,.sheet{overflow-wrap:anywhere;word-break:break-word}
.msg{max-height:none;overflow:visible}
table{display:block;overflow-x:auto;white-space:normal}
td{word-break:break-word}
.hdr{position:static;z-index:auto}
@media (max-width:768px){
  .block-container{padding-left:.6rem;padding-right:.6rem}
  .iphone{width:100%;max-width:300px}
  .pamt{font-size:1.5rem}
  .kpi-v{font-size:1.5rem}
  .hdr{flex-direction:column;align-items:flex-start}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading engine...")
def load_all():
    cfg = load_config()
    rows = [json.loads(l) for l in (ROOT / "data/raw/transactions.jsonl").open(encoding="utf-8")]
    ft = {s.code: list(s.message_templates) for s in SCAM_TYPES}
    eng = build_engine_from_data(cfg, rows, ft)
    return cfg, rows, eng


def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


cfg, ROWS, ENG = load_all()
METRICS = load_json(ROOT / "data/processed/metrics.json")
ADAPT = load_json(ROOT / "data/processed/adaptivity.json")
MODEL = load_json(ROOT / "data/processed/models/behavioural_model.json")
HELPLINE = cfg.get("demo", {}).get("helpline", "1930")
BLOCK_T = cfg["gate"]["block_threshold"]
STEP_T = cfg["gate"]["stepup_threshold"]

# ------------------------------------------------------------- header
st.markdown(
    '<div class="hdr"><div><div class="logo">FRAUD<b>RAKSHAK</b>'
    '<span class="live">● LIVE DEMO</span></div>'
    '<div class="tech">Intelligent UPI Fraud Detection</div></div>'
    '<div style="text-align:right"><div class="tech"><span class="dot"></span>SYSTEM NOMINAL · '
    'ENGINE v1.0</div><div class="tech" style="margin-top:.3rem">Demonstration only — '
    'not a banking authorization service · not connected to any live bank</div></div></div>',
    unsafe_allow_html=True)

# ------------------------------------------------------------- session state
if "verdict" not in st.session_state:
    st.session_state.verdict = None
if "txn" not in st.session_state:
    st.session_state.txn = None

by_type: dict[str, list[dict]] = {}
for r in ROWS:
    by_type.setdefault(r["type_code"], []).append(r)
labels = {s.code: f"{s.code} — {s.label} ⛔" for s in SCAM_TYPES}
for r in ROWS:
    if r["label"] == "legit":
        labels.setdefault(r["type_code"], f"{r['type_code']} — {r['type_label']} ✅")

# ------------------------------------------------------------- main workspace
colL, colR = st.columns([35, 65])

with colL:
    st.markdown('<div class="panel"><div class="tech">Payment testing</div><h3>TEST A TRANSACTION</h3>',
                unsafe_allow_html=True)
    choice = st.selectbox("Scenario (10 fraud F01–F10 · 10 genuine)",
                          sorted(by_type), format_func=lambda c: labels.get(c, c))
    idx = st.select_slider("Example", options=range(len(by_type[choice])),
                           format_func=lambda i: f"#{i+1} of {len(by_type[choice])}")
    t = by_type[choice][idx]
    is_fraud = t["label"] == "fraud"
    st.markdown(
        f'<div class="tech">{html.escape(t["type_code"])} · '
        f'{"<span style=color:#F97066>FRAUD SAMPLE</span>" if is_fraud else "<span style=color:#32D583>GENUINE SAMPLE</span>"}</div>'
        f'<div style="font-size:1.6rem;font-weight:800">₹{t["amount"]:,}</div>'
        f'<div class="kv"><span>Direction</span><b>{html.escape(t["direction"])}</b></div>'
        f'<div class="kv"><span>Type</span><b>{html.escape(t["type_label"][:32])}</b></div>'
        f'<div class="msg" style="margin-top:.5rem">“{html.escape(t["message"])}”</div>',
        unsafe_allow_html=True)
    if st.button("⚡ SCORE THIS PAYMENT"):
        v = ENG.score(t)  # real pipeline
        st.session_state.verdict = v
        st.session_state.txn = t
    v = st.session_state.verdict
    if v is not None:
        dc = "#F04438" if v.action == "BLOCK" else ("#F79009" if v.action == "STEP_UP" else "#12B76A")
        st.markdown(
            f'<div class="tech" style="margin-top:.8rem">Decision · risk</div>'
            f'<div style="font-size:1.5rem;font-weight:800;color:{dc}">{v.action} · {v.risk:.0%}</div>',
            unsafe_allow_html=True)
        st.markdown('<div class="tech">Four layers (actual engine output)</div>', unsafe_allow_html=True)
        for name, s in v.layers.items():
            pct = max(0.0, min(1.0, s)) * 100
            st.markdown(
                f'<div class="lyr"><div class="nm">{html.escape(name)}</div>'
                f'<div class="tr"><div class="fl" style="width:{pct:.0f}%"></div></div>'
                f'<div class="vl">{s:.2f}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="tech">Reasons</div>', unsafe_allow_html=True)
        for r in v.reasons:
            tag = html.escape(str(r.get("layer", "rule")).upper())
            st.markdown(f'<div class="ev e-block">[{tag}] {html.escape(r["text"])}</div>',
                        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with colR:
    st.markdown('<div class="panel"><div class="tech">User device preview</div><h3>WHAT THE USER SEES</h3>',
                unsafe_allow_html=True)
    pt = st.session_state.txn or t
    pv = st.session_state.verdict
    # radar ring: lime arc sized by actual risk
    risk = pv.risk if pv else 0.0
    arc = risk * 360
    act = pv.action if pv else None
    ring_col = "#F04438" if act == "BLOCK" else ("#F79009" if act == "STEP_UP" else "#C6FF00")
    sheet = ""
    if pv is not None:
        if pv.action == "BLOCK":
            cls, bdg = "s-block", '<span class="bdg b-block">⛔ Payment blocked</span>'
            title = "You are about to PAY — not receive!"
            sub = (f"This message claims money is arriving, but approving sends "
                   f"<b>₹{pt['amount']:,}</b> OUT of your account. Do not approve.")
            evc = "e-block"
        elif pv.action == "STEP_UP":
            cls, bdg = "s-step", '<span class="bdg b-step">⚠️ Verify first</span>'
            title = "Extra verification needed"
            sub = ("This payment needs one more check. Verify the recipient in your "
                   "official payment app or over a trusted channel before paying.")
            evc = "e-step"
        else:
            cls, bdg = "s-allow", '<span class="bdg b-allow">✅ No block</span>'
            title = "No fraud signals detected"
            sub = ("Known payee pattern and ordinary context. Stay alert — this is "
                   "a risk assessment, not a safety guarantee.")
            evc = "e-allow"
        evs = "".join(f'<div class="ev {evc}">{html.escape(r["text"])}</div>'
                      for r in pv.reasons[:4])
        if not evs:
            evs = '<div class="ev e-allow">No risk signals fired on this transaction.</div>'
        sheet = (f'<div class="sheet {cls}">{bdg}<div style="font-weight:800;margin:.35rem 0">'
                 f'{title}</div><div style="font-size:.74rem;color:#D6D9D2;line-height:1.5">{sub}</div>'
                 f'<div style="display:flex;justify-content:space-between;font-size:.7rem;margin-top:.4rem">'
                 f'<span>RISK</span><b style="color:{ring_col}">{risk:.0%}</b></div>'
                 f'<div class="rbar"><div style="width:{risk*100:.0f}%;background:{ring_col}"></div></div>'
                 f'{evs}<div class="hl">Suspect fraud? Call <b>{HELPLINE}</b> · cybercrime.gov.in</div></div>')
    st.markdown(
        f'<div class="iphone"><div class="screen"><div class="island"></div>'
        f'<div class="statusbar"><span>9:41</span><span>●●● ▮ 5G ▮▮▮</span></div>'
        f'<div class="riskring">'
        f'<svg width="104" height="104" viewBox="0 0 120 120">'
        f'<circle cx="60" cy="60" r="52" fill="none" stroke="#1E221D" stroke-width="7"/>'
        f'<circle cx="60" cy="60" r="52" fill="none" stroke="{ring_col}" stroke-width="7" '
        f'stroke-linecap="round" stroke-dasharray="{arc:.0f} 400" transform="rotate(-90 60 60)"/>'
        f'<text x="60" y="68" text-anchor="middle" fill="#F5F5F5" font-size="22" font-weight="800">'
        f'{risk:.0%}</text></svg></div>'
        f'<div class="phead">'
        f'<div class="tech">{html.escape(pt["direction"])} · {html.escape(pt["type_code"])}</div>'
        f'<div style="font-weight:700">{html.escape(pt["payee_name"])}</div>'
        f'<div class="tech">{html.escape(pt["payee_vpa"])}</div>'
        f'<div class="pamt">₹{pt["amount"]:,}</div></div>'
        f'<div class="pbody"><div class="msg">“{html.escape(pt["message"])}”</div>'
        f'<div class="kv"><span>Payee</span><b>{html.escape(pt["payee_name"][:22])}</b></div>'
        f'<div class="kv"><span>UPI ID</span><b>{html.escape(pt["payee_vpa"])}</b></div>'
        f'<div class="kv"><span>Amount</span><b>₹{pt["amount"]:,}</b></div>'
        f'{sheet}</div><div class="homebar"></div></div></div>',
        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------- analyst intel
st.markdown('<div class="panel"><div class="tech">Full-width intelligence</div>'
            '<h3>ANALYST INTELLIGENCE</h3>'
            '<div class="tech">Precomputed on the labelled synthetic set · thresholds '
            f'BLOCK ≥ {BLOCK_T} · STEP-UP ≥ {STEP_T}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="tech">A · Key performance indicators (from metrics.json)</div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
if METRICS:
    o = METRICS["overall"]
    kpis = [
        (k1, f"{METRICS['n_transactions']:,}", "Transactions scored", "labelled synthetic set"),
        (k2, f"{o['recall']:.1%}", "Fraud recall", f"tp {o['tp']} · fn {o['fn']}"),
        (k3, f"{o['false_positive_rate']:.2%}", "False-positive rate", f"fp {o['fp']} of {o['fp']+o['tn']:,} genuine"),
        (k4, f"{METRICS['latency_ms_per_txn']:.2f} <small>ms</small>", "Scoring latency", "mean per transaction"),
    ]
else:
    kpis = [(k1, "N/A", "Transactions scored", "metrics.json missing"),
            (k2, "N/A", "Fraud recall", ""), (k3, "N/A", "False-positive rate", ""),
            (k4, "N/A", "Scoring latency", "")]
for col, val, lab, sub in kpis:
    col.markdown(f'<div class="panel"><div class="kpi-v">{val}</div>'
                 f'<div class="tech">{lab}</div><div class="tech">{sub}</div></div>',
                 unsafe_allow_html=True)

st.markdown('<div class="tech">B · Model intelligence — 18 behavioural features (actual fitted weights)</div>',
            unsafe_allow_html=True)
if MODEL:
    rows = []
    for f, w in zip(MODEL["feature_names"], MODEL["w"]):
        d = "▲ risk" if w > 0.05 else ("▼ protective" if w < -0.05 else "— neutral")
        cls = "pos" if w > 0.05 else ("neg" if w < -0.05 else "")
        rows.append(f'<tr><td><code>{html.escape(f)}</code></td>'
                    f'<td>{html.escape(FEATURE_REASON.get(f, ""))}</td>'
                    f'<td style="font-family:monospace">{w:+.3f}</td>'
                    f'<td class="{cls}">{d}</td></tr>')
    st.markdown(f'<div class="panel"><table><tr><th>Feature</th><th>Plain English</th>'
                f'<th>Weight</th><th>Direction</th></tr>{"".join(rows)}</table>'
                f'<div class="tech" style="margin-top:.5rem">Linear model: a coefficient alone '
                f'does not explain any single prediction; reasons come from the engine.</div></div>',
                unsafe_allow_html=True)
else:
    st.warning("behavioural_model.json missing — run train.py.")

st.markdown('<div class="tech">C · Mule-ring network — top 5 (live graph query)</div>', unsafe_allow_html=True)
rings = ENG.graph.top_rings(5)
cards = []
for i, r in enumerate(rings, 1):
    n = min(r["size"], 12)
    nodes = "".join(
        f'<circle cx="{60 + 34 * np.cos(2 * np.pi * j / max(n, 1)):.0f}" '
        f'cy="{42 + 34 * np.sin(2 * np.pi * j / max(n, 1)):.0f}" r="6" '
        f'fill="{"#F04438" if j < r["flagged"] else "#1E221D"}" stroke="#252A24"/>'
        for j in range(n))
    cards.append(
        f'<div class="panel"><div class="tech">RING-{i:02d}</div>'
        f'<svg width="120" height="84" viewBox="0 0 120 84">{nodes}</svg>'
        f'<div style="font-size:1.3rem;font-weight:800">{r["size"]} <span class="tech">accounts</span></div>'
        f'<div class="tech"><span style="color:#F97066">{r["flagged"]} flagged</span> · '
        f'shares device / phone / IP with blocked accounts</div></div>')
cc = st.columns(5)
for col, card in zip(cc, cards):
    col.markdown(card, unsafe_allow_html=True)

st.markdown('<div class="tech">D · Adaptivity — the key judging point (from adaptivity.json)</div>',
            unsafe_allow_html=True)
if ADAPT:
    b, a = ADAPT["adversarial_recall_before_adaptation"], ADAPT["adversarial_recall_after_adaptation"]
    d = ADAPT["recovery_percentage_points"]
    st.markdown(
        f'<div class="panel" style="border-color:#C6FF00">'
        f'<div style="display:flex;gap:2rem;flex-wrap:wrap">'
        f'<div><div class="tech">Clean baseline</div><div class="kpi-v">{ADAPT["clean_fraud_recall"]:.1%}</div></div>'
        f'<div><div class="tech">Under structural attack</div>'
        f'<div class="kpi-v" style="color:#F97066">{b:.1%}</div></div>'
        f'<div><div class="tech">After analyst review + retrain</div>'
        f'<div class="kpi-v" style="color:#C6FF00">{a:.1%}</div></div>'
        f'<div><div class="tech">Recovery</div><div class="kpi-v" style="color:#C6FF00">+{d:.1f} pp</div></div>'
        f'</div><div class="tech" style="margin-top:.6rem">Attack: typos + Hinglish rewrites, aged payee, '
        f'familiar device, modest amount, no velocity ({ADAPT["misses_before_adaptation"]} misses reviewed). '
        f'Honest cost: false positives {ADAPT["legit_fpr_before_adaptation"]:.2%} → '
        f'{ADAPT["legit_fpr_after_adaptation"]:.2%}. Structural evasion is built to look genuine — '
        f'some of that cost is irreducible.</div></div>', unsafe_allow_html=True)
else:
    st.warning("adaptivity.json missing — run mutate.py.")

st.markdown(
    '<div class="tech" style="text-align:center;padding:1rem">Bounce'D · synthetic data only · '
    'cannot run inside a real UPI app · helpline 1930 · cybercrime.gov.in</div>',
    unsafe_allow_html=True)
