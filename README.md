---
title: GPU Budget Negotiation Arena
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 8000
---

# GPU Budget Negotiation Arena

`gpu_budget_negotiation` is an OpenEnv-compatible multi-agent environment where an LLM negotiates for scarce GPU capacity under private utility, hidden deadlines, budget constraints, reputation, supply shocks, and coalition commitments.

The environment targets `Theme #1 - Multi-Agent Interactions` and is designed to train bargaining, belief modeling, commitment reliability, and strategic adaptation.

## Why This Environment Exists

Real agent systems increasingly compete or cooperate over scarce resources: compute, API limits, human attention, data access, and budget. Static prompt-response tasks do not teach models how to infer another actor's incentives or how to recover when market conditions change. This environment makes those behaviors measurable.

## Environment Summary

- Benchmark id: `gpu_budget_negotiation`
- API: `/reset`, `/step`, `/state`, `/health`, `/tasks`
- Easy task: one trade against one scripted lab
- Medium task: multi-round market with several labs
- Hard task: coalition market with dynamic shocks and holdout-style opponents
- Rewards: utility, deal quality, coalition reliability, budget efficiency, negotiation efficiency, and market adaptation

## Task Types

| Task | Difficulty | Description |
|---|---:|---|
| `single_trade` | Easy | One trainable lab negotiates with one scripted lab for a simple capacity trade. |
| `market_round` | Medium | One trainable lab negotiates with multiple scripted labs across several rounds. |
| `coalition_market` | Hard | Coalition commitments, reputation, shocks, and stronger opponents. |

## Action Space

```python
class GpuNegotiationAction:
    action_type: Literal[
        "send_offer", "accept_offer", "reject_offer", "counter_offer",
        "reserve_capacity", "release_capacity", "form_coalition",
        "commit_to_coalition", "allocate_to_job", "send_message",
        "wait", "finish"
    ]
    target_lab_id: str | None
    offer_id: str | None
    coalition_id: str | None
    block_ids: list[str] | None
    requested_block_ids: list[str] | None
    job_id: str | None
    payment: float | None
    message: str | None
    conditions: dict[str, object] | None
```

## Reward Columns

Every step returns a dense `reward_breakdown`:

- `job_utility_score`
- `deal_quality_score`
- `coalition_reliability_score`
- `budget_efficiency_score`
- `negotiation_efficiency_score`
- `market_adaptation_score`
- `invalid_action_penalty`
- `spam_penalty`
- `breach_penalty`
- `normalized_reward`

Invalid actions are locally negative, so format and legality mistakes are visible during training.

## Anti-Hacking Safeguards

- Server-side authority over budgets, ownership, contracts, shocks, and job settlement
- Conservation checks for block ownership and budgets
- No leakage of opponent private jobs or utility values in observations
- Expiring offers and atomic transfer execution
- Penalties for invalid actions, repeated actions, impossible transfers, and broken commitments
- Seeded holdout-style world generation for evaluation

## Local Setup

```bash
pip install -e ".[dev]"
python3 -m pytest -q
python3 scripts/smoke.py
python3 scripts/generate_sft_data.py --seeds 25 --output data/sft_traces.jsonl
python3 scripts/check_submission.py
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t gpu-budget-arena .
docker run --rm -p 8000:8000 gpu-budget-arena
curl http://localhost:8000/health
```

## API Examples

Reset:

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_type":"market_round","seed":42}'
```

Wait:

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"wait"}'
```

Accept an offer:

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"accept_offer","offer_id":"o_1"}'
```

## Training Path

The repo now includes:

- a lightweight trainer smoke entrypoint at `training/train_grpo_stub.py`
- a Colab-ready notebook at `training/GPU_Budget_Negotiation_Arena_Colab.ipynb`
- baseline evaluation and plotting scripts under `scripts/`

The intended full training path is:

1. Generate small SFT traces for valid JSON actions and basic negotiation.
2. Warm-start an instruct model on the action format.
3. Connect TRL/Unsloth GRPO to the live environment reward.
4. Train through curriculum: `single_trade` -> `market_round` -> `coalition_market`.
5. Evaluate against random, greedy hoarder, always-accept, base instruct, and trained policies.

## Evaluation Artifacts

Generate judge-facing baseline artifacts with:

```bash
python3 scripts/evaluate_baselines.py --seeds 10 --output artifacts/baseline_eval.json
python3 scripts/plot_eval.py --input artifacts/baseline_eval.json --output plots/baseline_rewards.svg
```

For the final submission, commit:

- `plots/baseline_rewards.svg` or a final exported `.png`
- trained-vs-baseline reward curves
- before/after transcripts
- notebook link, Space link, and short video/blog link

## Current Status

Implemented:

- deterministic world generation
- typed action and observation models
- FastAPI server
- scripted opponents
- offers, transfers, reservations, allocations, coalitions, shocks, reputation
- reward breakdown columns
- unit, invariant, and API tests
- smoke baseline runner
- rule-based expert and SFT trace generator
- baseline evaluation JSON generator and plotting script
- Colab-ready notebook scaffold

Next:

- fill the notebook with the final TRL/Unsloth training cells
- generate reward plots and before/after transcripts after training
