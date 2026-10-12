# UPIGuard

**Pre-payment fraud guard for UPI** — scores a payment *before money moves* and warns the payer with reasons they can act on.

CyberCup 6.0 · Problem CC6-AI02 — Intelligent Detection of Creative Payment Frauds

---

## The idea in one line

UPI fraud is not a transaction anomaly. The amount, time and payee can all look normal. It is a **social-engineering script** — someone convinces you that money is *arriving* when the button you press sends money *out*.

The most common variant inverts the `Collect Request`: *"Your refund of ₹4,999 has been credited — accept the request."* It reads as receiving money. It is a **debit**. The whole system is built to catch that class of lie.

## What it does

A four-layer cascade scores each transaction, because different fraud families leave different evidence:

| Layer | Catches | Method |
|---|---|---|
| Rule engine | High-precision obvious patterns | Deterministic checks over `config.yaml` |
| Script similarity | Reused scam scripts | TF-IDF cosine, pure Python, char 3-grams for typo tolerance |
| Behavioural model | Payee / amount / velocity / device | L2 logistic regression over 18 named features |
| Graph layer | Mule rings, organised fraud | Account–identifier graph, bounded BFS |

Fused into one probability, then a three-way gate:

- **BLOCK** (≥ 0.72) — stop, show reasons + safe actions + helpline 1930
- **STEP_UP** (≥ 0.38) — allow, but demand OTP re-authentication in-app
- **ALLOW** — proceed

The middle state is deliberate. Blocking everything and blocking nothing both score zero; only a calibrated system is useful.

## Results (8,000 synthetic transactions)

| | |
|---|---|
| Recall | 100.0% *(in-distribution — see below)* |
| Precision | 99.45% |
| **False positive rate** | **0.12%** |
| Latency | 0.13 ms/transaction |

**Under adversarial attack** (text evasion + structural evasion — aged payee, familiar device, low velocity, modest amount):

| Stage | Recall |
|---|---|
| Clean | 100.0% |
| Text-only attack | 98.5% |
| **Structural attack** | **46.5%** |
| After analyst review + retrain | 98.8% (+52.3 pp recovery) |

The 100% in-distribution figure is **not a generalisation claim** — our rules and TF-IDF index share a taxonomy with the generator, so they recognise their own training scripts. The structural-attack row is the number that means something. The full write-up, including why the adaptation recovery costs 0.12% → 6.71% false positives, is in [`docs/REPORT.md`](docs/REPORT.md).

## Run it

```bash
pip install -r requirements.txt     # numpy, pandas, pyyaml, streamlit

python3 src/generator/generate.py   # 8,000 labelled transactions
python3 src/detector/train.py       # train + metrics + self-checks
python3 src/detector/mutate.py      # attack + adaptation loop

streamlit run app/main.py           # the demo, 3 screens
```

Screens: **Payment Simulation** (the gate firing), **Adaptivity** (attack → review → recover), **Analyst Dashboard** (drill-down, ring graph, feature weights).

## Why no scikit-learn / LightGBM / NetworkX

The risk engine is `numpy` + stdlib only. A gradient-boosted tree would score marginally better on this data, but:

1. The linear model's weights **are** the reason codes — the alert can say "first payment to a 6-day-old payee" without a SHAP dependency.
2. It trains in under a second on CPU, which is what makes the analyst feedback loop a live demo instead of a batch job.

Fewer dependencies also means a judge can run this with one `pip install`.

## Correctness note

The graph layer is seeded from accounts the **rule layer** blocked — replayed over history — never from the ground-truth `label` field. Seeding from labels would let it "discover" fraud using the answer key and would make every reported number meaningless. See `engine.seed_from_blocks`.

## Repo layout

```
src/generator/   taxonomy.py (10 fraud + 10 legit types) · generate.py (labelled data + fraud rings)
src/detector/    rules.py · similarity.py · features.py · model.py · graph.py · engine.py
                 train.py (metrics + asserts) · mutate.py (adversarial + adaptation)
app/main.py      Streamlit demo
configs/         config.yaml — thresholds, fusion weights, rule scores
docs/REPORT.md   full write-up, prototype scope, limitations
```

`train.py` and `mutate.py` self-check with assertions and fail loudly on regression — recall below 85% or FPR above 15% will not pass silently.
