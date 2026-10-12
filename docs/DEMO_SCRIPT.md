# UPIGuard — demo script (4 minutes)

Rehearse this five times with a timer. The demo fails on nerves far more often
than on software.

---

## 0:00 — Legit flow first (10 sec)

Screen ① → pick **L03 Friend dinner split** → *Run the guard*.

> "Dinner at Cafe Blue — my share Rs 600."

**Expect: ALLOW, risk ~11%.**

*Say:* "Firche ek normal payment. Green. Low friction — kyunki system ko rok
nahi karna, sirf fraud rokna hai."

---

## 1:00 — The headline catch (60 sec)

Screen ① → pick **F01 Collect request disguised as refund** → *Run the guard*.

> "Your refund of Rs 4,999 has been credited. Kindly accept the request."

**Expect: BLOCK.** Full-screen red, reasons stacked:
- refund/credit wording + collect request → money goes OUT
- message closely matches known refund-scripts
- first-ever payment, payee account a few days old

*Say:* "Yeh screen ki ek line poori class rok rahi hai — `COLLECT_REQUEST`
matlab paisa ja raha hai. User ko lagta hai refund aa raha hai."

Then scroll to **Layer scores**: rules high, behavioural contributing.

---

## 2:00 — Calibrated, not trigger-happy (30 sec)

Screen ① → pick **F10 Fake job** → *Run the guard*.

**Expect: STEP_UP** (risk ~66%, just under the block line).

*Say:* "Har cheez block nahi karta. Yeh medium risk hai — OTP re-auth maango,
phir jaane do. System calibrated hai."

---

## 2:30 — The ring (30 sec)

Screen ③ Analyst Dashboard → **Connected fraud rings**.

Largest: 148 accounts, 95 flagged.

*Say:* "Ye 148 accounts individually bilkul clean lagte hain. Sab ek device
fingerprint aur ek QR merchant se jude hain. Ek hi query se 95 accounts flag —
ye koi per-transaction detector nahi kar sakta."

---

## 3:00 — Adaptivity, honestly (60 sec)

Screen ② Adaptivity.

Four numbers: 100% → 98.5% → **46.5%** → 98.8%.

*Say:* "Ye honest number hai. Adversary ne script badli — typos, Hinglish, naye
shabd — aur structural tells bhi hataye: payee ka account purana kar diya, device
pehchan wala rakh diya, amount chhota kar diya, velocity zero. System **tod
gaya**, 46.5% pe gir gaya."

> "Ab analyst ne 771 misses review kiye, naye scripts corpus me daale, model
> retrain kiya. 98.8%. **+52.3 points recovery.**"

Then point at the last column:

*Say:* "Aur ye dekho — recovery free nahi hai. False positives 0.12% se 6.71%
ho gaye. Kyunki structural-evasion wala row *banaya* hi genuine payment jaisa
hai. Ye partly irreducible hai — hum chhupate nahi, report karte hain."

---

## 4:00 — Close on the ask (20 sec)

*Say:* "UPIGuard payment ke *pehle* rokta hai, wajah batata hai, aur naye scam
pe girta hai — phir khud seekhta hai. 0.13 milliseconds per transaction, real
payment intent ke andar budget ke andar."

**Stop. Don't keep talking.**

---

## Rehearse these three failure modes

| If this happens | Do this |
|---|---|
| Laptop hangs on load | Screen 2 tab already open; go to ③ first. Backup video (`docs/`) is your fallback. |
| A judge says "100% recall is suspicious" | **Agree immediately.** "Correct — in-distribution hai, generator aur detector same taxonomy use karte hain. Defensible number 46.5% → 98.8% wala hai." |
| A judge says "false positives 6.71% is too high" | "Correct, aur wahi tradeoff hai. Isliye gate teen-state hai — 6.7% pe bhi STEP_UP se zyada kuch block nahi hota. Production me isko labelled real traffic pe retune karna hoga." |

Both failure modes are answered by conceding and then pointing at the number
that shows you already know the weakness. **That is the whole strategy** — you
found the weakness before they did.

---

## Pre-demo checklist (15 min before)

- [ ] `python3 src/generator/generate.py && python3 src/detector/train.py && python3 src/detector/mutate.py` — all three exit clean
- [ ] `streamlit run app/main.py` — loads in under 5 seconds
- [ ] All 3 tabs open, correct scenarios pre-selected
- [ ] Browser zoom at 100%, font readable from 2 metres
- [ ] Backup video open in a second tab, ready
- [ ] Laptop on AC, notifications off, one browser profile only
