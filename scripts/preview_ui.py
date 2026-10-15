"""Render the app's verdict UI to a static HTML file, for visual review.

The Streamlit app itself cannot be screenshotted from this environment, but the
HTML it produces is plain strings plus one CSS block. This script pulls the real
CSS out of app/main.py, runs the real engine on real scenarios, and writes a
preview that shows exactly what the browser will render.

Run:  python3 scripts/preview_ui.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "detector"))
sys.path.insert(0, str(ROOT / "src" / "generator"))

from engine import build_engine_from_data, load_config  # noqa: E402
from taxonomy import SCAM_TYPES  # noqa: E402

OUT = Path("/tmp/ui_preview.html")

# Pull the live CSS out of the app so the preview can never drift from it.
app_src = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
css = re.search(r'CSS = """(.*?)"""', app_src, re.S).group(1)

cfg = load_config()
rows = [json.loads(l) for l in (ROOT / "data" / "raw" / "transactions.jsonl").open(encoding="utf-8")]
eng = build_engine_from_data(cfg, rows, {s.code: list(s.message_templates) for s in SCAM_TYPES})


def pick(code: str) -> dict:
    return next(r for r in rows if r["type_code"] == code)


SAFE = {
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
}


def render_case(code: str, heading: str) -> str:
    t = pick(code)
    v = eng.score(t)
    cls = {"BLOCK": "v-block", "STEP_UP": "v-stepup", "ALLOW": "v-allow"}[v.action]
    head, kicker = {
        "BLOCK": ("Payment stopped", "blocked at the gate"),
        "STEP_UP": ("Extra check needed", "held for verification"),
        "ALLOW": ("Payment cleared", "passed the gate"),
    }[v.action]

    reasons = "".join(
        f'<div class="reason">{r["text"]}<br><span class="src">{r.get("layer","rule").upper()} LAYER</span></div>'
        for r in v.reasons) or '<div class="reason">No risk signals fired.</div>'
    safe = "".join(f'<div class="safe">{s}</div>' for s in SAFE[v.action])
    layers = "".join(
        f'<div class="layer"><div class="nm">{n}</div>'
        f'<div class="track"><div class="fill" style="width:{min(1,s)*100:.0f}%"></div></div>'
        f'<div class="vl">{s:.2f}</div></div>' for n, s in v.layers.items())

    return f'''
<section style="margin-bottom:3.5rem">
  <div class="sectionlabel">{heading}</div>
  <div class="scenario">
    <div class="meta">{t["direction"]} · {t["type_code"]}</div>
    <div class="msg">"{t["message"]}"</div>
    <div class="payee">{t["payee_name"]} · {t["payee_vpa"]} · Rs {t["amount"]:,}</div>
  </div>
  <div class="verdict {cls}">
    <div style="display:flex;justify-content:space-between;align-items:flex-end;gap:1.5rem">
      <div>
        <div class="kicker">{kicker}</div>
        <div class="headline">{head}</div>
        <div class="detail">Rs {t["amount"]:,} · scored before money moves</div>
      </div>
      <div class="riskbox">
        <div class="num">{v.risk*100:.0f}<span style="font-size:1.4rem">%</span></div>
        <div class="cap">risk score</div>
      </div>
    </div>
  </div>
  <div class="sectionlabel">What the guard saw</div>
  {reasons}
  <div class="sectionlabel">What to do instead</div>
  {safe}
  <div class="sectionlabel">How the four layers scored</div>
  {layers}
</section>'''


rings = eng.graph.top_rings(3)
ring_html = "".join(
    f'<div class="ring"><div class="size">{r["size"]}</div>'
    f'<div class="desc">accounts acting as one operation · '
    f'<b style="color:var(--danger)">{r["flagged"]}</b> already flagged<br>'
    f'Individually clean. Suspicious together.</div></div>' for r in rings)

html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">{css}</head>
<body class="stApp">
<div class="block-container">
  <h1 style="margin-bottom:.3rem">🛡️ BounceD</h1>
  <p style="color:var(--muted);margin-bottom:2rem">Pre-payment fraud guard for UPI · visual preview</p>
  {render_case("L03", "Legitimate payment — should pass")}
  {render_case("F01", "Refund collect-request — the headline catch")}
  {render_case("F10", "Fake job — should be held, not blocked")}
  <section>
    <div class="sectionlabel">Connected fraud rings</div>
    {ring_html}
  </section>
</div>
</body></html>'''

OUT.write_text(html, encoding="utf-8")
print(f"wrote {OUT} ({len(html)} bytes)")
print("cases: L03 legit, F01 refund scam, F10 fake job")
