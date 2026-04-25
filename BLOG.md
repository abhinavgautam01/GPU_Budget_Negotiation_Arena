# GPU Budget Negotiation Arena: Training Agents to Bargain for Compute

Modern AI systems increasingly compete for scarce resources: GPU hours, API quotas, data access, and human attention. Static QA benchmarks do not train an agent to reason about another actor's incentives, deadlines, hidden utility, or willingness to cooperate. GPU Budget Negotiation Arena turns that problem into an OpenEnv-compatible multi-agent training environment.

One trainable lab negotiates against scripted opponent labs for GPU blocks. Each lab has private jobs, hidden deadlines, budgets, reputations, and reliability constraints. The agent can send offers, counter-offers, reserve capacity, allocate blocks to jobs, form coalitions, and submit natural-language pitches. The server owns all state transitions, so agents cannot directly mutate budgets, ownership, jobs, shocks, or contracts.

The environment has three curriculum tasks:

- `single_trade`: one controlled lab negotiates with one scripted lab.
- `market_round`: multiple labs negotiate across several rounds.
- `coalition_market`: hard-mode negotiation with commitments, shocks, reputation, and holdout-style opponents.

The reward is intentionally decomposed. Every step returns job utility, deal quality, coalition reliability, budget efficiency, negotiation efficiency, market adaptation, invalid-action penalties, spam penalties, breach penalties, and normalized reward. This makes failure modes visible during training rather than hiding them in a single scalar.

The latest version adds a hybrid judge-agent extension. The deterministic environment reward remains primary, but `judge_mode="rule"` lets a trainable lab make a natural-language GPU allocation pitch. Opponent labs generate adaptive counter-pitches from their own private needs, and a frozen local judge scores urgency, evidence, reliability, fairness, and coalition value. This gives a demo-friendly language layer without making nondeterministic LLM judging the only reward source.

For reproducible training proof, the repo includes a lightweight REINFORCE-style policy selector. It learns which negotiation strategy to use for each curriculum stage and emits a real reward curve. In the generated artifacts, evaluation reward starts near the random baseline and rises as the selector learns valid allocation, trade acceptance, and coalition behavior. A before/after transcript shows the untrained policy mostly rejecting offers and waiting, while the trained selector accepts useful trades, allocates blocks to private jobs, and forms coalitions.

The same environment can be connected to SFT or GRPO. The repo generates SFT traces and chat-format JSONL, and the Colab notebook includes optional Unsloth/TRL cells for replacing the lightweight selector with model-weight fine-tuning.

Key artifacts:

- `plots/baseline_rewards.svg`: matplotlib reward progress curve with bot baselines, expert ceiling, and judge-bonus trend.
- `artifacts/training_report.md`: final reward table and selected trained policies.
- `artifacts/before_after_training.md`: same-seed qualitative before/after transcript.
- `artifacts/judged_transcript.md`: natural-language judged negotiation transcript.

The core result is that the environment is not just a resource-allocation simulator. It creates pressure for theory-of-mind behavior: the agent must infer incentives from partial observations, decide when to trade versus hoard, preserve reputation, and use coalitions when market shocks make solo optimization brittle.
