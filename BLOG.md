# GPU Budget Negotiation Arena: Training a Llama to Bargain for Compute

Modern AI agents don't just answer questions — they compete for compute, API
budget, data access, and human attention. There's no benchmark for that. We
built one, and trained Llama-3.2-3B end-to-end on it: SFT + GRPO against the
live environment reward. Two of three task means flipped from negative to
positive.

## The setup

Five fictional AI labs share a single GPU pool just before a paper deadline.
Each lab has private jobs with hidden value, urgency, reliability requirements,
and budget. They send offers, counter-offers, reservations, allocations,
coalition commitments, and (with the optional judge layer) natural-language
pitches arguing for priority. The trainable lab is `lab_0`. The server is
authoritative throughout — agents cannot mutate balances, ownership, jobs,
contracts, or shocks directly. The action layer is typed Pydantic, the state
machine is enforced in `gpu_budget_arena/env.py`, and a five-file pytest suite
guards the invariants.

Three curriculum tasks: `single_trade` (one-on-one), `market_round`
(multi-lab, multi-round), and `coalition_market` (hard-mode with capacity,
energy, reliability, and demand shocks plus reputation effects).

Reward is decomposed into eleven components — job utility, deal quality,
coalition reliability, judge argument score, budget efficiency, negotiation
efficiency, market adaptation, plus four explicit penalties (invalid action,
local repeated action, breach, spam) — so failure modes are visible during
training instead of hiding inside a single scalar. That decomposition is what
makes "always accept" an honest opponent rather than a degenerate strategy:
the spam and breach penalties cap its ceiling.

## What was actually trained

**Stage 1 — Unsloth SFT.** `unsloth/Llama-3.2-3B-Instruct` fine-tuned on
expert-replay chat traces (`data/sft_messages.jsonl`) for 500 steps over 13
epochs. Loss fell `1.5356 → 0.0196` (98.7% drop). The trainer's own
`log_history` is in `artifacts/sft_training_curve.json`; the live Space plots
it directly. The run survived a mid-training crash and was resumed from
`checkpoint-120`, which is why the LR schedule on the dashboard shows two
cosine phases stitched together.

**Stage 2 — TRL GRPO against the live environment reward.** The SFT'd LoRA was
warm-started into `trl.GRPOTrainer` with a custom reward function that
literally calls `GpuBudgetNegotiationEnv.step(action).reward + format_bonus`
on each sampled completion (parse penalty if the JSON is malformed). 300
training steps × 4 completions per step over 85 replayed env observations
spanning every task. Per-batch mean reward climbed `0.031 → 0.1595` with a
peak of `0.233` mid-run. The full curve is in
`artifacts/grpo_training_curve.json`. There is no surrogate reward, no learned
reward model, and no proxy — every gradient comes from the env's own scalar.

**Stage 3 — Live policy rollout vs every scripted baseline.** The SFT model
was wrapped via `gpu_budget_arena/llm_policy.py`, plugged in as `lab_0`, and
rolled out across 5 seeds × 3 tasks against six scripted baselines and the
untrained base Llama. Per-task results:

| Task | Base Llama | SFT-trained Llama | Δ |
|---|---:|---:|---:|
| `single_trade` | `0.0495` | `0.1642` | `+0.115` (3.3×) |
| `market_round` | `−0.0505` | `0.1890` | `+0.240` (sign flip) |
| `coalition_market` | `−0.2805` | `0.4162` | `+0.697` (sign flip) |
| **overall** | `−0.0938` | `+0.2565` | `+0.350` |

That puts the trained Llama 3rd of 8 evaluated policies — ahead of every
scripted bot except the always-accept opportunist (`0.266` overall) and the
hand-authored rule expert (`0.451`, the structural ceiling). Always-accept
beats us on a single scalar but loses unambiguously on `coalition_market`,
under the judge layer, and on coalition reliability. The full discussion is
in `JUDGE_QA.md` Q4.

## The hybrid judge layer

A frozen rule-based judge (`gpu_budget_arena/judge.py`) scores natural-language
pitches on five axes: urgency, evidence, reliability, fairness, and coalition
value. Crucially, the deterministic env reward stays primary — the judge
contributes a bounded bonus, never the dominant signal. This is the inverse
of most LLM-judge benchmarks where nondeterministic judging is the only reward
source. The 20-round judged-negotiation transcript in
`artifacts/judged_transcript.md` shows `lab_0` overtaking the deadline-pressured
`lab_2` at round 6 and winning 14 of 20 rounds.

## Reproducibility

The lightweight CPU baseline (REINFORCE-style policy selector, `0.1217 →
0.4490` over 180 episodes) reproduces with `python3 scripts/check_submission.py`
in roughly 60 seconds, no GPU required. The full SFT + GRPO pipeline runs in
about an hour on a free Colab T4 from
`training/GPU_Budget_Negotiation_Arena_Colab.ipynb`. Every artifact the live
Space renders is committed: `artifacts/sft_training_curve.json`,
`artifacts/grpo_training_curve.json`, `artifacts/trained_llm_summary.json`,
`artifacts/before_after_training.md`, `artifacts/judged_transcript.md`,
`plots/baseline_rewards.svg`, `plots/sft_loss_curve.svg`,
`plots/grpo_reward_curve.svg`, `plots/trained_llm_vs_baselines.svg`,
`plots/training_dashboard.svg`. The 5-file pytest suite enforces the
server-authoritative state contract end-to-end.

## What's next

1. Persist the GRPO LoRA across Colab sessions and run it as a live policy
   to add the fourth column (`trained_llm_grpo`) to the comparison bar chart.
2. Add OpenAI / Anthropic / Gemini API adapters as alternative `lab_0`
   policies for direct frontier comparison.
3. Train a small learned reward model on episode-level judge scores so GRPO
   can optimise pitch quality, not just env reward.

The core argument the project makes is that this isn't a resource-allocation
toy. It creates a bench where theory-of-mind behaviour is measurable: the
agent has to infer incentives from partial observations, decide when to trade
versus hoard, preserve reputation, and lean on coalitions when shocks make
solo optimisation brittle. The trained Llama doesn't do all of that
perfectly — but the gradient is real, the numbers are real, and the
environment is honest enough to surface where it fails.
