# Architecture

## System flow

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
  │ config.  │  │  similarity  │  │     model      │  │  (rings)   │
  │ yaml     │  │  TF-IDF cos  │  │  logistic reg. │  │ bounded    │
  │          │  │  + 3-grams   │  │  18 features   │  │ BFS        │
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

## The central catch

The system stores payment **direction** as a first-class field, because the dominant scam is a direction mismatch:

| Message says | Button actually is | Result |
|---|---|---|
| "Your refund of ₹4,999 is credited — accept" | `COLLECT_REQUEST` | money leaves the account |

A user reads "refund / credited" and expects an inflow. The rule `collect_request_refund` fires on exactly this pairing, and the warning screen names the gap: *"Message promises a refund, but a collect request sends money OUT."*

## Graph seeds are not labels

```
                  ┌─────────────┐
  transaction ───▶│ rule engine │──┐
                  └─────────────┘  │
                                    ▼
                            blocked accounts
                                    │
                                    ▼
                  ┌─────────────────────────────┐
                  │ RingDetector seeded from    │
                  │ rule blocks — NEVER from    │
                  │ ground-truth labels         │
                  └─────────────────────────────┘
                                    │
                  device / phone / IP ─┴─▶ connected components
                                    │
                                    ▼
                  "clean" account flagged by ring membership
```

Seeding from `label` would let the graph discover fraud using the answer key. `engine.seed_from_blocks` replays the rule layer over history instead, which is what a deployed system does with its blocklist.

## Three screens

| Screen | Purpose | Judges see |
|---|---|---|
| ① Payment Simulation | The gate firing on a chosen scenario | BLOCK / STEP_UP / ALLOW with reasons and safe actions |
| ② Adaptivity | Attack → review → recover | 100% → 46.5% → 98.8%, and the false-positive cost |
| ③ Analyst Dashboard | Interrogation | Drill-down, ring list, learned feature weights |

## Dependency boundary

```
┌─────────────────────────────────────────────┐
│  app/main.py          streamlit (only pip) │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  src/detector/      numpy + stdlib only    │
│  rules · similarity · features · model ·   │
│  graph · engine · train · mutate           │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  src/generator/     stdlib only             │
│  taxonomy · generate                       │
└─────────────────────────────────────────────┘
```

The risk engine runs with no ML stack installed, so a judge can run the detector from a bare Python install. Only the UI needs `pip install`.
