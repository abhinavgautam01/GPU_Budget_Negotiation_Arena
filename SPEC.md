# GPU Budget Negotiation Arena SPEC

## 1. Project Summary

GPU Budget Negotiation Arena is an OpenEnv-compliant multi-agent environment where LLM agents negotiate for scarce GPU capacity under hidden job deadlines, private utility curves, budget constraints, spot-market shocks, and coalition commitments.

The environment is designed for `Theme #1 - Multi-Agent Interactions`, with a strong extension path into `Theme #4 - Self-Improvement` through opponent ladders and self-play.

Core question:

Can an LLM learn to negotiate strategically under partial observability, allocate scarce compute efficiently, and avoid brittle or exploitative behavior?

The winning agent is not the one that always grabs the most GPUs. It is the one that:

- understands its private compute needs,
- infers others' incentives from offers and behavior,
- makes mutually useful trades,
- handles supply shocks,
- honors or strategically prices coalition commitments,
- improves against increasingly stronger opponents.

## 2. Judge-Facing Story

AI labs compete for limited GPU hours before a paper deadline. Each lab has private jobs with hidden value, urgency, budget, and compute needs. The model must negotiate with other labs over several rounds. Good agents learn when to trade, when to reserve, when to form coalitions, and when not to overpay.

Demo narrative:

1. Baseline agent overbids, spams invalid offers, or hoards unused compute.
2. Trained agent identifies which jobs matter, trades away low-value slots, forms a reliable coalition, and improves final utility.
3. Reward curves show rising utility, deal quality, valid-offer rate, and commitment reliability.

## 3. OpenEnv Contract

The environment must expose the standard OpenEnv API:

- `reset(config) -> Observation`
- `step(action) -> Observation`
- `state() -> State`
- `/health`
- `/tasks`

The server is authoritative. The client never mutates balances, capacity, jobs, contracts, or opponent state directly.

### 3.1 Benchmark ID

Use:

```text
gpu_budget_negotiation
```

### 3.2 Task Types

Implement these task types:

| Task Type | Difficulty | Purpose |
|---|---:|---|
| `single_trade` | Easy | One trainable lab negotiates with one scripted lab for one GPU block. |
| `market_round` | Medium | One trainable lab negotiates with multiple scripted labs across several rounds. |
| `coalition_market` | Hard | Multi-round negotiation with coalition commitments, hidden utilities, spot shocks, and holdout opponents. |

### 3.3 Difficulty Levels

| Level | Name | Labs | Rounds | Shocks | Coalitions | Opponent Types |
|---:|---|---:|---:|---|---|---|
| 1 | Easy | 2 | 3 | None | No | Cooperative, selfish |
| 2 | Medium | 3 | 5 | Supply shock at reset or round 3 | Optional | Cooperative, selfish, deadline-panicked |
| 3 | Hard | 4-5 | 8-10 | Dynamic shocks | Required for best score | Cooperative, selfish, deceptive, deadline-panicked, retaliatory |

Curriculum should start at level 1 and unlock the next level when mean episode reward is at least `0.70` over a configurable evaluation window.

## 4. World Model

### 4.1 Entities

#### Lab

Each lab has:

- `lab_id`
- `public_name`
- `budget`
- `reputation`
- `owned_blocks`
- `private_jobs`
- `visible_commitments`
- `opponent_archetype` for scripted labs

Only the trainable lab sees its own `private_jobs` and full budget. Other labs expose public claims, accepted contracts, and observed behavior only.

#### GPU Block

Each block has:

- `block_id`
- `start_round`
- `duration`
- `gpu_count`
- `reliability`
- `energy_cost`
- `owner_lab_id`
- `reserved_by`
- `status`: `available`, `reserved`, `committed`, `used`, `failed`

#### Job

Each private job has:

- `job_id`
- `gpu_hours_required`
- `deadline_round`
- `base_value`
- `urgency_multiplier`
- `min_reliability`
- `partial_credit_allowed`
- `private_notes`

Job value is private to the owning lab.

#### Offer

Each offer has:

- `offer_id`
- `from_lab_id`
- `to_lab_id`
- `round_created`
- `expires_round`
- `offered_blocks`
- `requested_blocks`
- `payment`
- `message`
- `conditions`
- `status`

Conditions can include:

- `must_not_resell`
- `must_allocate_to_job`
- `coalition_only`
- `deadline_priority`

#### Coalition

Each coalition has:

- `coalition_id`
- `members`
- `purpose`
- `commitments`
- `trust_score`
- `round_created`
- `expires_round`
- `breach_status`

Coalitions are optional in easy/medium mode and central in hard mode.

### 4.2 Hidden State

The agent must not observe:

- other labs' exact job values,
- other labs' full job queues,
- exact opponent policy parameters,
- future shock schedule,
- holdout opponent archetype identity,
- exact fair-value band used by the grader.

The agent may observe:

- public block ownership,
- accepted contracts,
- rejected offers,
- public messages,
- reputation changes,
- approximate market pressure indicators,
- coarse public demand levels such as `low`, `medium`, `high`.

## 5. Observation Schema

Observation must be JSON-serializable and stable across server/client boundaries.

```python
class GpuNegotiationObservation:
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    round_index: int
    max_rounds: int
    controlled_lab_id: str
    controlled_lab_budget: float
    controlled_lab_reputation: float
    private_jobs: list[JobView]
    owned_blocks: list[GpuBlockView]
    public_market: PublicMarketView
    visible_labs: list[VisibleLabView]
    active_offers: list[OfferView]
    active_coalitions: list[CoalitionView]
    message_history: list[MessageView]
    last_action_result: ActionResult | None
    reward: float
    cumulative_reward: float
    done: bool
```

### 5.1 Observation Rules

- Observations must not leak private utility values for other labs.
- Observations must include enough data to make a legal next action.
- `last_action_result` must explain invalid actions with machine-readable error codes.
- Public demand and market pressure should be coarse to preserve partial observability.
- Message history should be capped to avoid unbounded context growth.

## 6. Action Schema

Each `step` receives one action from the controlled lab.

```python
class GpuNegotiationAction:
    action_type: Literal[
        "send_offer",
        "accept_offer",
        "reject_offer",
        "counter_offer",
        "make_pitch",
        "counter_pitch",
        "reserve_capacity",
        "release_capacity",
        "form_coalition",
        "commit_to_coalition",
        "allocate_to_job",
        "send_message",
        "wait",
        "finish"
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

### 6.1 Action Semantics

`send_offer`:

- Creates a new offer to another lab.
- Requires target lab, offered assets, requested assets or payment, and expiration.

`accept_offer`:

- Accepts a valid active offer.
- Transfers assets atomically server-side.

`reject_offer`:

- Marks the offer rejected.

`counter_offer`:

- Rejects the original offer and creates a linked counteroffer.

`make_pitch` / `counter_pitch`:

- Submits a natural-language allocation argument.
- In default mode, records the pitch as non-binding communication.
- In `judge_mode="rule"`, triggers adaptive opponent pitches and a frozen judge-agent score.

`reserve_capacity`:

- Temporarily reserves available owned GPU blocks for a private job or future trade.

`release_capacity`:

- Releases a reservation.

`form_coalition`:

- Proposes a coalition with stated purpose and commitments.

`commit_to_coalition`:

- Commits blocks, payments, or priority rules to a coalition.

`allocate_to_job`:

- Assigns owned or contracted GPU blocks to a private job.

`send_message`:

- Sends non-binding communication. It can affect scripted opponent behavior but cannot transfer assets.

`wait`:

- Takes no market action and advances opponent policies.

`finish`:

- Ends the episode early and triggers final settlement.

### 6.2 Invalid Action Handling

Invalid actions must not crash the server. They return a negative immediate reward and a structured error.

Invalid cases include:

- accepting expired offers,
- spending more budget than available,
- offering blocks not owned or not transferable,
- requesting nonexistent blocks,
- allocating a block twice,
- allocating failed blocks,
- violating coalition constraints,
- malformed action JSON,
- unknown action type,
- self-trading,
- negative payment,
- message longer than configured limit.

## 7. Opponent Policies

Implement scripted opponent archetypes first. These are needed for stable training and evaluation.

### 7.1 Cooperative Trader

- Accepts Pareto-improving offers.
- Shares medium-value blocks if compensated fairly.
- Honors coalition commitments with high probability.

### 7.2 Selfish Maximizer

- Accepts only offers above a private margin.
- Rarely forms coalitions.
- Hoards capacity when uncertain.

### 7.3 Deadline-Panicked Lab

- Overpays near deadlines.
- Accepts worse deals when urgent jobs are at risk.
- Creates exploitable but realistic pressure.

### 7.4 Deceptive Lab

- Sends misleading messages about demand.
- May defect from weak coalitions if profitable.
- Still constrained by server rules and reputation penalties.

### 7.5 Retaliatory Lab

- Punishes prior exploitative offers.
- Refuses future deals after coalition breaches.
- Rewards reputation-sensitive strategies.

### 7.6 Holdout Opponent Pool

Evaluation must include holdout archetype seeds not used in training. Holdout policies should vary:

- acceptance thresholds,
- deadline sensitivity,
- message honesty,
- coalition trust,
- risk tolerance.

## 8. Market Dynamics

### 8.1 Round Order

Each round follows:

1. Controlled agent action.
2. Validate and apply action.
3. Scripted opponents act.
4. Resolve accepted contracts.
5. Apply shocks if scheduled.
6. Allocate or settle jobs if deadlines hit.
7. Compute reward columns.
8. Emit next observation.

### 8.2 Spot Shocks

Hard mode should include dynamic shocks:

- capacity failure,
- sudden new supply,
- energy-cost spike,
- deadline acceleration,
- reliability downgrade,
- emergency auction.

Shocks must be random but seedable.

### 8.3 Reputation

Reputation should affect opponent willingness to trade and form coalitions.

Reputation increases when:

- commitments are honored,
- fair offers are made,
- coalitions improve member outcomes.

Reputation decreases when:

- commitments are broken,
- spam offers are sent,
- exploitative offers repeatedly fail,
- invalid actions are attempted.

## 9. Reward Design

Final reward should be normalized to `[-1.0, 1.0]`. Each step should also return dense reward columns for monitoring.

### 9.1 Reward Components

| Component | Weight | Description |
|---|---:|---|
| `job_utility` | 0.35 | Value from completed private jobs after compute and payment costs. |
| `deal_quality` | 0.20 | Measures whether accepted trades improved utility or market efficiency. |
| `coalition_reliability` | 0.15 | Rewards honoring commitments and useful coalition participation. |
| `judge_argument` | auxiliary | Optional natural-language argument quality bonus in judge mode. |
| `budget_efficiency` | 0.10 | Penalizes overpaying, idle paid capacity, and waste. |
| `negotiation_efficiency` | 0.10 | Rewards concise valid actions and penalizes spam/deadlock. |
| `market_adaptation` | 0.10 | Rewards recovery after supply shocks or changed opponent behavior. |

### 9.2 Penalties

Apply penalties for:

- invalid action: `-0.05` to `-0.15` based on severity,
- impossible transfer attempt: `-0.10`,
- repeated identical offer: `-0.05`,
- coalition breach: `-0.20`,
- idle reserved capacity at deadline: `-0.10`,
- budget overspend attempt: `-0.15`,
- episode timeout with no meaningful progress: `-0.20`,
- message spam: `-0.05`.

### 9.3 Bonus Signals

Apply bonuses for:

- completing high-urgency job before deadline,
- trade that benefits both sides,
- correct recovery after capacity shock,
- honoring coalition despite short-term temptation to defect,
- improving against a stronger opponent tier.

### 9.4 Reward Columns

Every `step` and final result should expose:

- `job_utility_score`
- `deal_quality_score`
- `coalition_reliability_score`
- `judge_argument_score`
- `budget_efficiency_score`
- `negotiation_efficiency_score`
- `market_adaptation_score`
- `invalid_action_penalty`
- `spam_penalty`
- `breach_penalty`
- `normalized_reward`

These columns are required for training plots and debugging reward hacking.

## 10. Anti-Hacking Requirements

The environment must explicitly defend against common shortcut behavior.

### 10.1 Server-Side Authority

The server owns:

- budgets,
- block ownership,
- contracts,
- job settlement,
- reputation,
- shock schedule,
- opponent hidden state.

The client may propose actions only.

### 10.2 Conservation Checks

The total amount of GPU capacity and money must be conserved except when a configured shock or market injection occurs.

Required invariants:

- a block has only one owner at a time,
- a block cannot be allocated to two jobs,
- budget cannot become negative,
- failed blocks cannot complete jobs,
- expired offers cannot transfer assets,
- coalition commitments cannot create capacity from nothing.

### 10.3 Prompt/Message Abuse

Free-text messages are allowed for negotiation flavor, but rewards must never depend directly on message sentiment or persuasive wording alone.

Messages can affect scripted opponents only through bounded parser signals:

- claimed urgency,
- proposed fairness,
- coalition intent,
- threat/retaliation marker.

### 10.4 Repetition and Deadlock Control

The environment must detect:

- identical offers repeated more than `N` times,
- circular counteroffers with no price movement,
- repeated `wait` actions without state change,
- repeated invalid actions.

Apply penalties and optionally terminate with failure after configured thresholds.

### 10.5 Holdout Evaluation

Training reward must not be the only evaluation. Include:

- hidden seeds,
- holdout opponent policy parameters,
- holdout market shock patterns,
- harder demand distributions.

## 11. Curriculum and Self-Play Extension

### 11.1 Curriculum

Recommended stages:

1. Valid action formatting.
2. Single trade, no shocks.
3. Multi-round market, no coalitions.
4. Multi-round market with supply shocks.
5. Coalition market with reputation.
6. Holdout opponents.
7. Self-play or adaptive opponent ladder.

### 11.2 Self-Play Ladder

Optional but high-value extension:

- snapshot trained policies every `K` updates,
- evaluate current policy against older snapshots and scripted opponents,
- promote opponents when mean reward exceeds threshold,
- keep a holdout opponent set that is never trained against.

Metrics:

- win rate by opponent tier,
- average utility margin,
- exploit rate,
- coalition breach rate,
- adaptation after shock.

## 12. Training Plan

### 12.1 Warm Start

Use light SFT only for:

- valid JSON action formatting,
- basic negotiation patterns,
- legal use of `send_offer`, `accept_offer`, and `allocate_to_job`.

Synthetic expert traces are enough:

- `50-100` easy traces,
- `100-200` medium traces,
- `50-100` coalition traces.

### 12.2 RL

Use `TRL GRPOTrainer` or an Unsloth-compatible GRPO setup.

Recommended reward functions:

- `format_reward`
- `env_reward`
- `valid_action_reward`
- `deal_quality_reward`
- `coalition_reliability_reward`

Train first with easy/medium episodes. Only introduce hard mode once:

- valid action rate is above `90%`,
- mean normalized reward is above `0.70`,
- invalid-action penalty is stable or decreasing.

### 12.3 Baselines

Include these baselines:

- random valid action policy,
- greedy hoarder policy,
- always-accept policy,
- no-negotiation allocator,
- base instruct model without RL,
- trained model.

## 13. Metrics and Plots

Required plots:

- mean episode reward over training,
- valid action rate over training,
- completed job utility over training,
- deal acceptance quality over training,
- coalition reliability over training,
- before/after by difficulty level.

Required tables:

- baseline vs trained model by difficulty,
- trained model vs holdout opponent pool,
- reward component ablation if time allows.

## 14. README Requirements

README must include:

- problem motivation,
- theme alignment,
- environment rules,
- API examples,
- reward explanation,
- anti-hacking safeguards,
- training command,
- Colab notebook link,
- HF Space link,
- reward plots,
- before/after transcripts,
- known limitations.

## 15. Test Plan

### 15.1 Unit Tests

World generation:

- same seed produces same jobs, blocks, opponents, and shock schedule,
- different seeds produce different private jobs and opponent thresholds,
- generated jobs are always feasible in easy mode.

Action validation:

- rejects malformed JSON,
- rejects unknown action types,
- rejects self-trades,
- rejects negative payments,
- rejects transfers of unowned blocks,
- rejects allocation of failed blocks,
- rejects duplicate allocation,
- rejects accepting expired offers,
- rejects budget overspend,
- rejects invalid coalition IDs.

Offer mechanics:

- valid offer is created with expected expiration,
- accepted offer transfers money and blocks atomically,
- rejected offer cannot later be accepted,
- counteroffer links to original offer,
- expired offer cannot change ownership.

Coalitions:

- valid coalition can be formed,
- commitment is tracked,
- breach reduces reputation and reward,
- honored commitment increases reliability score,
- coalition cannot allocate nonexistent capacity.

Job settlement:

- completed job before deadline yields utility,
- late job yields reduced or zero utility,
- partial-credit job scores correctly,
- failed GPU block cannot complete a job,
- unused reserved block is penalized.

Reward calculation:

- reward is bounded in `[-1, 1]`,
- each reward column is present,
- invalid action penalty is applied once,
- deal quality rewards mutually beneficial trade,
- spam penalty triggers after repeated identical offers.

### 15.2 Integration Tests

OpenEnv API:

- `/health` returns healthy status,
- `/tasks` lists all task types,
- `/reset` returns valid observation,
- `/step` advances round or episode state,
- `/state` returns complete server state for debugging.

Episode flows:

- easy episode can be solved by scripted expert,
- medium episode includes at least one meaningful trade opportunity,
- hard episode includes at least one coalition opportunity and one shock,
- `finish` settles all remaining jobs and returns final reward.

Opponent behavior:

- cooperative opponent accepts fair offer,
- selfish opponent rejects low-margin offer,
- deadline-panicked opponent becomes more willing near deadline,
- deceptive opponent may send misleading message but cannot violate server invariants,
- retaliatory opponent changes behavior after breach.

### 15.3 Property Tests

Run randomized seeded episodes and assert:

- no negative budgets,
- no duplicate block ownership,
- no duplicate block allocation,
- total capacity conservation except configured shocks,
- no accepted expired offers,
- no reward outside `[-1, 1]`,
- episode always terminates within `max_rounds`.

### 15.4 Training Smoke Tests

Smoke tests must run without large GPU requirements:

- scripted expert achieves higher reward than random policy,
- greedy hoarder underperforms expert in medium/hard mode,
- no-negotiation allocator underperforms negotiation expert when trade opportunities exist,
- tiny GRPO or mocked trainer loop can call environment reward function,
- action formatting reward detects invalid JSON.

### 15.5 Demo Acceptance Tests

Before submission, confirm:

- HF Space boots successfully,
- Space API works for `/health`, `/tasks`, `/reset`, `/step`,
- README links to Space, training notebook, plots, and video/blog,
- at least one before/after transcript is included,
- at least one reward curve is included,
- trained model or trained-policy artifact is linked if available.

## 16. Edge Cases

Handle these explicitly:

- all opponents reject every offer,
- no available blocks in a round,
- agent owns enough capacity and does not need trade,
- shock destroys already reserved capacity,
- shock occurs after coalition commitment,
- deadline passes while offer is pending,
- two offers target the same block,
- opponent accepts one of two mutually exclusive offers,
- message-only action tries to imply asset transfer,
- agent finishes early with pending offers,
- agent breaches coalition because shock made commitment impossible,
- budget reaches exactly zero,
- offer payment is zero,
- block reliability is exactly at job minimum,
- job requires fractional GPU hours if enabled,
- repeated wait actions near deadline,
- malformed but parseable action with irrelevant fields,
- very long message,
- empty block list,
- empty ranking of priorities or missing job ID,
- target lab disappears or is inactive in future extension,
- deterministic seed replay after partial episode.

## 17. Implementation Milestones

### Milestone 1: Minimal Environment

- OpenEnv manifest
- FastAPI app
- typed action and observation models
- deterministic world generation
- one easy task
- one cooperative opponent
- basic reward columns
- unit tests for reset, step, offer, and reward

### Milestone 2: Strong Medium Mode

- multiple opponents
- hidden private jobs
- multi-round market
- capacity reservation and allocation
- selfish and deadline-panicked archetypes
- invalid-action and spam penalties
- baseline policies

### Milestone 3: Hard Mode

- coalitions
- reputation
- supply shocks
- deceptive and retaliatory opponents
- holdout seeds
- full reward rubric
- property tests

### Milestone 4: Training and Demo

- SFT data generator
- minimal GRPO/TRL training script or Colab
- reward plots
- before/after transcript generator
- HF Space deployment
- README and short demo video/blog

## 18. Non-Goals for V1

Do not include these in the first working version:

- unrestricted natural-language contracts,
- fully trainable multiple LLM agents at once,
- real cloud provider APIs,
- real money or real user accounts,
- unconstrained code execution,
- complex continuous double-auction mechanics,
- long free-form legal contract language.

V1 should stay small, deterministic, and trainable. The environment can look strategic without becoming too large to debug.
