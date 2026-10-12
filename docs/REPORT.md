# UPIGuard — Pre-Payment Fraud Guard for UPI

**CyberCup 6.0 · Problem CC6-AI02 — Intelligent Detection of Creative Payment Frauds**
Team UPIGuard · ASET, Amity School of Engineering & Technology

---

## 1. Problem

UPI fraud in India is not a transaction anomaly — the amount, time and payee can all look perfectly normal. It is a **social-engineering script**: someone convinces the victim that money is *arriving*, when the button they press sends money *out*.

The most common variant inverts the UPI `Collect Request`. "Your refund of ₹4,999 has been credited — accept the request" reads as receiving money. It is a **debit**. The victim accepts, and their balance drops.

Two properties make this hard to detect, and they shaped every design decision:

1. **The text lies about intent.** Any detector keyed on words like "refund" learns the wrong thing. The signal is the *mismatch* between stated intent and payment direction.
2. **Fraud is organised.** A mule ring runs dozens of accounts from one device fingerprint and one QR merchant. Every account looks clean in isolation. No per-transaction detector can see it; the pattern can.

**Our goal:** score the transaction *before money moves*, and warn the payer with reasons they can act on.

## 2. Approach

A four-layer cascade, because different fraud families leave different evidence. Layers are independent so each can be reported, tuned, and turned off separately.

| Layer | Catches | Method | Why this method |
|---|---|---|---|
| **Rule engine** | Obvious, high-precision patterns | Deterministic checks over `config.yaml` | Every hit is explainable; analyst can edit rules without a deploy |
| **Script similarity** | Reused scam scripts | TF-IDF + cosine, pure Python, with char 3-grams | Fraud is a script; 3-grams give typo tolerance without a stemmer |
| **Behavioural model** | Payee / amount / velocity / device patterns | L2 logistic regression, hand-built features, numpy | Every weight *is* a reason code — no SHAP dependency needed |
| **Graph layer** | Mule rings and organised fraud | Account–identifier graph, bounded BFS from flagged seeds | Finds accounts that are individually clean and collectively guilty |

Scores are fused by a weighted sum into a calibrated probability, then passed through a three-way gate:

- `BLOCK` (≥ 0.72) — stop, show reasons and safe actions
- `STEP_UP` (≥ 0.38) — allow, but demand OTP re-authentication in-app
- `ALLOW` (< 0.38) — proceed

The middle state matters. A system that blocks everything and a system that blocks nothing both score zero; only a *calibrated* system is useful. We report false-positive rate for exactly this reason.

**On the graph layer's seeds.** The graph is seeded from accounts the *rule layer* blocked, replayed over history — never from the ground-truth label. Seeding from labels would let the graph "discover" fraud using the answer key, and every number in this report would be meaningless. This is implemented in `engine.seed_from_blocks` and is the single most important correctness property in the codebase.

## 3. Architecture

```
                    ┌──────────────────────────────────────┐
  Transaction ─────▶│ Intent Parser                        │
  (UPI request)     │  direction, purpose, payee, QR, ctx  │
                    └───────────────┬──────────────────────┘
                                    │
        ┌───────────────┬───────────┴────────┬────────────────┐
        ▼               ▼                    ▼                ▼
  ┌──────────┐  ┌──────────────┐  ┌────────────────┐  ┌────────────┐
  │  Rules   │  │   Script     │  │  Behavioural   │  │   Graph    │
  │          │  │  similarity  │  │     model      │  │  (rings)   │
  │ config.  │  │  TF-IDF cos  │  │  logistic reg. │  │ bounded    │
  │ yaml     │  │  + 3-grams   │  │  18 features   │  │ BFS        │
  └────┬─────┘  └──────┬───────┘  └────────┬───────┘  └─────┬──────┘
       │               │                   │                 │
       └───────────────┴─────────┬─────────┴─────────────────┘
                                 ▼
                    ┌────────────────────────┐
                    │ Risk fusion (weighted) │
                    │   + sigmoid → 0..1     │
                    └───────────┬────────────┘
                                ▼
                    ┌────────────────────────┐
                    │ Pre-payment gate       │
                    │ BLOCK / STEP_UP / ALLOW│
                    └───────────┬────────────┘
                                ▼
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌───────────────────┐                        ┌────────────────────┐
│ Warning screen    │                        │ Analyst dashboard  │
│ reasons + safe    │                        │ drill-down, rings, │
│ actions + 1930    │                        │ feature weights,   │
└───────────────────┘                        │ analyst → retrain  │
                                             └────────────────────┘
```

**Dependency choice.** The risk engine is `numpy` + stdlib only — no scikit-learn, no LightGBM, no NetworkX. This was a deliberate trade: a gradient-boosted tree would score marginally better on this data, but the linear model's weights are directly readable as reason codes, and the whole thing trains in under a second on CPU, which is what makes the analyst feedback loop a live demo rather than a batch job. Only the Streamlit front-end needs a pip install.

## 4. Results

Dataset: 8,000 synthetic transactions, 1,440 fraud (18%), 6,560 genuine. Reproduce with `python3 src/generator/generate.py`.

### 4.1 Detection performance (in-distribution)

| Metric | Value |
|---|---|
| Recall (fraud caught) | 100.0% |
| Precision | 99.45% |
| F1 | 0.997 |
| **False positive rate** | **0.12%** (8 of 6,560 genuine) |
| Latency per transaction | 0.13 ms |

Per-fraud-family recall: 100% for all ten families. Highest-risk genuine traffic:

| Genuine type | FPR |
|---|---|
| Medical emergency (uses urgency language, new device — a deliberate trap) | 0.59% |
| Rent transfer | 0.31% |
| Insurance premium | 0.15% |
| Friend dinner split | 0.15% |

### 4.2 Read this before quoting §4.1

The 100% figure is **in-distribution and should not be read as a generalisation claim.** Our rules and TF-IDF index are built from the same 10-family taxonomy that generated the corpus, so they are recognising their own training scripts. A judge who tries one unseen scam will break it. The number that means something is §4.3.

We also found and fixed a real leak during development: the graph layer originally seeded its flagged set from the ground-truth `label` field, which would have let it "discover" fraud using the answer key. It now seeds from rule-layer blocks only.

### 4.3 Adversarial robustness — the honest result

We attack the system the way a competent adversary would, in two stages:

- **Stage 1 — text evasion.** Typos, Hinglish code-mixing ("please kro"), keyword substitution, brand look-alikes ("HDfC BnK", "lRCTC"), full paraphrase.
- **Stage 2 — structural evasion.** Remove the behavioural tells *as well*: age the payee account to 400–3,000 days, move onto a device and IP the user has used before, drop the amount to ₹400–3,000, kill the velocity, hide it as a single collect request.

| Stage | Fraud recall | Genuine FPR |
|---|---|---|
| Clean | 100.0% | 0.12% |
| After text-only attack | 98.5% | 0.12% |
| **After structural attack** | **46.5%** | 0.12% |
| After analyst review + retrain | 98.8% | 6.71% |

**Reading this honestly.** Text evasion barely dents the system, because the behavioural layer does not key on words. Structural evasion halves it — the system genuinely breaks, and we show that rather than hide it. Analyst review recovers **+52.3 points** of recall. The catch is the last column: recovery is *not* free, and false positives move 0.12% → 6.71%.

That cost is partly irreducible, and we would rather say so than tune it away. A structural-evasion row is *constructed* to resemble a genuine payment — established payee, familiar device, modest amount, no velocity. No feature-based system separates it from a real low-risk transfer. This is a property of the problem, not a bug we can retrain out, and it belongs in §6.

Per-family recall under attack shows which mechanisms the attacker beats:

| Family | Recall under attack |
|---|---|
| F02 QR code swap | 12% |
| F10 Fake job / task | 14% |
| F08 Lottery / prize | 22% |
| F09 Investment scheme | 37% |
| F05 Mule chain | 40% |
| F03 Screen-sharing | 98% |

### 4.4 Latency

0.13 ms per transaction single-threaded on CPU, all four layers included. The dominant costs are the bounded graph BFS and the cached TF-IDF dot product. An earlier version was ~15× slower; the reverse account→identifier index and TF-IDF vector cache in the fix cut it to the figure above. Real-time enforcement (inside a payment intent) is comfortably within budget.

## 5. Prototype scope

Per the problem statement's requirement to state clearly what the prototype can and cannot do:

**This prototype CAN:**
- Score a simulated UPI transaction before money moves, and return BLOCK / STEP_UP / ALLOW
- Explain every decision in plain language with the specific evidence behind it
- Detect all 10 modelled fraud families, including multi-account mule rings
- Detect a clean-looking account purely from its membership in a flagged ring
- Adapt to newly-labelled scam variants through an analyst review loop
- Enforce step-up verification (OTP re-auth) as a middle state between allow and block
- Recommend concrete safe actions and surface the cybercrime helpline (1930)

**This prototype CANNOT:**
- Run inside a real UPI application or intercept live payment rails — no bank/NPCI integration
- See any data a real system would have that ours does not: real device attestation, SIM state, beneficiary bank-side KYC, or transaction history beyond the synthetic window
- Detect fraud where the payer knowingly initiates a transfer to a genuinely new payee with no textual cue — this is a deliberate, unaddressed gap (§6)
- Detect SIM-swap attacks unless they produce a device or IP change
- See insider threats — a compromised legitimate account behaving within its own history
- Guarantee low false-positive rates on real traffic: our 0.12% is measured against a synthetic genuine distribution and will be higher in production
- Replace human judgement. The STEP_UP state exists precisely because some cases cannot be resolved automatically.

## 6. Limitations

| # | Limitation | Why it matters |
|---|---|---|
| 1 | **All data is synthetic.** No real fraud labels are public, and using live data is out of scope. Absolute metrics will not transfer; relative comparisons between layers may. | Judges should treat our numbers as directional, not as a benchmark. |
| 2 | **In-distribution recall is inflated** (100%) by the shared taxonomy between generator and detector. §4.3 is the defensible figure. | Quoting §4.1 alone would be misleading, and we say so explicitly. |
| 3 | **Structural evasion halves recall** (100% → 46.5%) and adaptation costs 6.7% FPR. Partly irreducible. | The honest trade curve, not a solved problem. |
| 4 | **No ground-truth leakage**, but the graph layer still benefits from a curated seed blocklist. In production that blocklist is the real hard problem, not the BFS. | Our ring detection assumes good seeds exist. |
| 5 | **Single fraud family per transaction.** Real scam flows span multiple transactions (contact → lure → collect → launder). We model the endpoint, not the campaign. | A campaign-level attacker who drips 20 small transfers across 3 weeks stays under every velocity threshold. |
| 6 | **English-dominant text with Hinglish injected in testing.** Code-mixed and regional-language scripts are only partially covered. | A fully Hindi or Tamil script would evade the TF-IDF layer entirely. |
| 7 | **Synthetic labels are rule-derived**, so model and rules share an ontology. A human-labelled set would likely show more disagreement. | Our own adaptivity loop partially simulates this, but not fully. |
| 8 | **Weights in `config.yaml` were hand-tuned**, not optimised against a held-out set. | Threshold selection is not principled; a PR-curve sweep would be. |

## 7. Reproducing

```bash
pip install -r requirements.txt          # only numpy, pandas, pyyaml, streamlit

python3 src/generator/generate.py        # writes 8,000 labelled transactions
python3 src/detector/train.py            # trains model, writes metrics.json
python3 src/detector/mutate.py           # attack + adaptation, writes adaptivity.json
streamlit run app/main.py                # the demo
```

`train.py` and `mutate.py` both self-check with assertions and fail loudly on regression: `train.py` fails if recall drops below 85% or FPR exceeds 15%; `mutate.py` fails if adaptation *reduces* recall or pushes FPR past 15%. A change that breaks the system cannot pass silently.

## 8. Files

| Path | Purpose |
|---|---|
| `src/generator/taxonomy.py` | 10 fraud + 10 legitimate transaction types |
| `src/generator/generate.py` | Labelled synthetic dataset with fraud rings |
| `src/detector/rules.py` | Deterministic rule engine |
| `src/detector/similarity.py` | TF-IDF script similarity, pure Python |
| `src/detector/features.py` | 18 named features + reason-code strings |
| `src/detector/model.py` | Logistic regression, numpy |
| `src/detector/graph.py` | Mule-ring detection, bounded BFS |
| `src/detector/engine.py` | Orchestration, fusion, pre-payment gate |
| `src/detector/train.py` | Training + metrics + self-checks |
| `src/detector/mutate.py` | Adversarial mutation + adaptation loop |
| `app/main.py` | Streamlit demo, 3 screens |
| `configs/config.yaml` | Thresholds, fusion weights, rule scores |
| `data/processed/*.json` | Generated metrics — the tables quoted above |
