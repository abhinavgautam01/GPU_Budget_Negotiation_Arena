# What happens when you teach an LLM to fight for GPU hours?

Most demos show a model answering questions. This one shows a model **negotiating**—because the next wave of agent systems will not be isolated chatbots. They will share clusters, API budgets, and deadlines with *other* agents that have their own hidden goals.

**GPU Budget Negotiation Arena** is a small, fully implemented world where that future is stress-tested. Five fictional AI labs share one GPU pool the night before a paper deadline. You control one lab. Everyone else is scripted, greedy, or strategic in different ways. The environment does not let you cheat: budgets, block ownership, offers, and coalition deals are enforced on the server. The model can only send structured actions and, optionally, a natural-language pitch that a separate rule-based judge can score for persuasion.

We trained **Llama-3.2-3B** on this world—first with supervised fine-tuning on expert-style traces, then with **GRPO** (a policy-gradient method from HuggingFace TRL) where the *actual* environment reward, not a proxy, drives the gradient. The headline result is simple to say and hard to fake: on two of the three task tiers, the same base model that **lost** money on average ended up **positive** after SFT, and the overall score moved from about **−0.09** to **+0.26** when compared to the untrained instruct model under identical seeds and tasks.

This post is the plain-language story of that project: what we built, how we trained it, what the numbers mean, and where the sharp edges are.

---

## The problem we wanted to model

Classroom negotiation benchmarks often look like “write a contract in English.” That is useful, but it is also familiar: models have seen millions of contract-like documents. **Resource negotiation under private utility** is different. You do not see the other side’s true deadline or the exact value of their job. You see offers, refusals, and sometimes a coalition proposal—and you have to decide whether to hoard GPU blocks, share them, or commit to a joint plan that can fall apart if someone breaks a promise.

We wanted an environment where:

- **Strategic depth** grows across three curriculum stages: a simple bilateral trade, a multi-lab “market” round, and a “coalition market” with shocks, reputation, and harder coordination.
- **Gaming the score** is hard: we split reward into many named components (utility, deal quality, coalition reliability, budget efficiency, and several penalties) so a single cheap trick does not look like “winning” without being exposed elsewhere.
- **The server is the source of truth**: agents propose actions; the simulator validates and updates state. No prompt injection into “balance sheets.”

That design lives mostly in one serious module, `gpu_budget_arena/env.py`, with typed Pydantic actions and a reward breakdown you can read in the code or in the OpenEnv-style manifest.

---

## What you actually *do* in the arena

You are **lab_0**. On each turn you can do things that real negotiators do in miniature: send an offer, counter, reserve capacity, withdraw, allocate blocks to a job, try to form or honor a coalition, send free text, or wait. The action space is finite but not trivial—enough that “always say yes to everything” is a real baseline, and so is a hand-tuned rule expert that knows the mechanics.

The three task types ramp difficulty:

- **`single_trade`** — get your head around offers and one opponent.
- **`market_round`** — several labs, several rounds, more room to misread the table.
- **`coalition_market`** — the hard mode: capacity and reliability pressure, and incentives that make solo greed brittle.

We also added an optional **hybrid judge**: a small, frozen rule system that scores short *pitches* for urgency, evidence, reliability, fairness, and coalition value. Important detail: the core environment reward still dominates. The judge adds a *bounded* bonus. We did not want a world where a flaky LLM-judge is the only signal—that would be a different project.

---

## How we trained the model (for real)

We did not stop at a stub. The pipeline is **SFT first, then GRPO**, and both stages touch real LLM weights.

**Supervised fine-tuning (Unsloth, 4-bit).** We generated chat-formatted trajectories from the simulator—expert and scripted play rolled into a JSONL in the same “system / user / assistant” shape the model sees at inference. We fine-tuned a LoRA on top of `unsloth/Llama-3.2-3B-Instruct` for 500 steps over 13 epochs. Loss went from about **1.54** to about **0.02**. That is not “we drew a line on a whiteboard”—the curve is extracted from the actual trainer state and lives in the repo as `artifacts/sft_training_curve.json` (and a matching plot the Space can show). Like many Colab stories, the run also survived a hiccup: we resumed from an intermediate checkpoint once, so if you look at the learning-rate schedule on the dashboard, you can see a second cosine “segment.” That is not a bug; it is a real interrupted run, honestly represented.

**GRPO against the live environment.** For the second stage we used TRL’s `GRPOTrainer`. The trick is how reward is defined: for each training prompt, we replay the world to a specific observation, sample several completions from the model, parse each as a JSON action, and feed it through **`GpuBudgetNegotiationEnv.step`**. The reward is the same scalar the game uses in deployment (plus a small format bonus, minus a parse penalty for garbage). So we are not training against a second network that *approximates* the world—we are training against the world’s own function, for better or worse, up to the usual sampling noise.

Over 300 steps with four rollouts per step on dozens of fixed prompts, the *mean* per-batch reward rose from a low start (around **0.03**) to about **0.16** on average, with a higher spike mid-run. The JSON summary and the SVG curve are in `artifacts/grpo_training_curve.json` and `plots/grpo_reward_curve.svg`.

**Evaluation.** After SFT, we ran the model as a full policy: same seeds and tasks for the untrained base model and for our adapter, so the comparison is apples-to-apples. The bar chart in the repo (`plots/trained_llm_vs_baselines.svg`) and the table (`artifacts/trained_llm_summary.json`) are the evidence bundle.

---

## The numbers, in one breath

Comparing the **SFT adapter** to the **untrained** instruct model on the same rollouts (five seeds, three task types, full episodes):

- **`market_round`**: from slightly negative to clearly positive (a sign flip in mean reward).
- **`coalition_market`**: from strongly negative to strongly positive—this is the line we are proudest of, because the hard tier is where naive behavior dies.
- **Overall**: the trained model sits **third out of eight** policies we evaluated—behind a one-line “always accept” bot on a single average scalar and behind a hand-authored rule expert that was written to exploit the environment’s mechanics. It beats the random-ish strategies, the greedy baselines, and, crucially, the *same* base LLama before training.

The honest part: **always-accept** still edges us on one aggregate number. We do not hide that. It is a structural ceiling: say yes, grab whatever is offered, and you look good on a single headline average until you look at *which* subscores collapse—coalition breaking under stress, or zero credit under the pitch judge. The write-up in `JUDGE_QA.md` walks through that tradeoff line by line.

---

## What else ships besides training curves

- A **lightweight CPU training path** (selector over scripted policies) that proves the reward signal is learnable without a GPU—useful for CI and for intuition.
- **55 pytest tests** across the environment, baselines, API shape, and training artifacts so refactors do not quietly break the contract.
- A **Hugging Face Space** that does not need you to read JSON by hand: tables, SVG plots, and links into the key files, plus HTTP endpoints for reset/step if you want to talk to the env programmatically.
- A **~60 second** one-command script and an even shorter `instant_demo.py` that print headline numbers and run one real episode on CPU, so a reviewer can feel the simulator without a GPU.
- A **Colab notebook** that strings clone → data generation → SFT → eval → GRPO → final artifacts in one pass for anyone with a T4 and patience.

We keep large weight folders out of git on purpose: Spaces and GitHub are happier with code and metrics; the adapter typically lives on Google Drive (or your own storage) and you point the eval scripts at it.

---

## What we would do next

**More GRPO** (or a longer schedule) to chase down parse errors—the model still fails to emit valid JSON on a meaningful fraction of steps, and a fallback action is a blunt instrument. **Live inference on the Space** with an L4 so visitors can play one lab as the trained model without Colab. **Frontier API baselines** as another lab, if we want a clean comparison to the best public chat models. And maybe a *small* learned component for long-horizon pitch quality—*after* the core env reward is stable.

---

## The bottom line

GPU Budget Negotiation Arena is a bet that **resource contention between agents** deserves the same kind of rigor as question answering. We built a simulator with teeth, put an actual Llama through SFT and GRPO on top of it, and published the numbers and plots so you can argue with the results instead of the slide deck.

If you only remember one thing: **we did not train against a hand-wavy reward.** We trained against the same stepwise reward the environment uses in deployment. The model is not perfect, but the gradient is not fake—and the hard tier is where that distinction actually shows up.

*Live Space:* [abhinavgautam01/gpu-budget-negotiation-arena on Hugging Face](https://huggingface.co/spaces/abhinavgautam01/gpu-budget-negotiation-arena)  
*Code:* [GitHub: GPU_Budget_Negotiation_Arena](https://github.com/abhinavgautam01/GPU_Budget_Negotiation_Arena)  
*Deeper spec:* see `README.md` and `SPEC.md`; judge prep and Q&A: `JUDGE_QA.md`, `PITCH.md`, `MENTOR_PREP.md`.
