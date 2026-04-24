# OpenEnv Hackathon Idea Pack

Assumptions used throughout:
- Team size: `3-4`
- Compute: hackathon GPUs only
- Must ship: `OpenEnv` environment on HF Spaces, minimal `TRL` or `Unsloth` training script, visible reward improvement, strong README/demo story
- Bias: objective verifiers, short-to-medium horizon v1, optional light SFT for action formatting

Scoring legend:
- `Innovation`, `Story`, `Trainability`, `Reward` are scored `/10`
- `Weighted` = `0.4*Innovation + 0.3*Story + 0.2*Trainability + 0.1*Reward`
- `Risk` is qualitative and should be read as delivery risk, not research value

## Theme 1: Multi-Agent Interactions

### 1. GPU Budget Negotiation Arena
- Concept: Several labs negotiate for scarce GPU hours under hidden deadlines, budget caps, and utility curves.
- Why it fits: It directly tests bargaining, coalition formation, bluff detection, and partially observable incentives.
- Observation / actions: The agent sees its own jobs, partial public market state, prior messages, and offer history; it can message, propose trades, accept, reject, reserve, or form coalitions.
- Episode / termination: `5-10` negotiation rounds, then a final allocation and utility calculation.
- Reward / verifier: Utility from completed jobs, bonus for Pareto-improving deals, penalty for idle compute, invalid offers, and broken coalition commitments.
- Anti-hacking safeguards: Hidden utility generation, contract validation server-side, no direct access to others' private state, and replay checks to prevent message-template farming.
- Why judges may care: Strong theme fit, easy to explain, and very aligned with real multi-agent resource allocation.
- Demo story: Baseline agent over-asks and stalls; trained agent trades low-priority slots for urgent compute and reaches better global utility.
- Build difficulty: `Medium`
- Training feasibility: `High`; short episodes and structured rewards make GRPO/RLVR practical.
- Recommended stack: OpenEnv FastAPI env with JSON action schema; start with scripted opponents; `GRPO`; light SFT for action formatting only.
- 1-day prototype path: Implement utility generator, `offer/accept/reject` loop, one scripted selfish opponent, and reward columns for deal quality and task completion.
- Biggest risk: If scripted opponents are too dumb, behavior collapses into simple heuristics instead of real negotiation.

### 2. Disaster Relief Coalition Board
- Concept: Multiple responder agents with private inventories coordinate across flood-hit regions with incomplete situational awareness.
- Why it fits: It mixes cooperation, competition for scarce assets, and coalition formation under uncertainty.
- Observation / actions: Region alerts, local inventory, partial map, incoming requests, and ally promises; actions include move, share inventory, request help, commit support, and broadcast claims.
- Episode / termination: `8-15` turns until all crises expire or time runs out.
- Reward / verifier: Saved-population score, penalty for duplicated response, late delivery, false commitments, and stranded resources.
- Anti-hacking safeguards: Hidden incident realizations, stochastic road closures, server-side conservation checks, and commitment tracking.
- Why judges may care: High emotional stakes and clear visual demo potential.
- Demo story: Untrained agents swarm the same city; trained policy learns coverage and role specialization.
- Build difficulty: `Medium-High`
- Training feasibility: `Medium`; longer horizon, but reward decomposition is still manageable.
- Recommended stack: OpenEnv grid/state env; start with one trainable agent plus scripted peers; `GRPO` with dense intermediate logistics rewards.
- 1-day prototype path: Make `3` regions, `2` asset types, private needs, and one coalition mechanic.
- Biggest risk: Too many moving parts can make early reward noisy.

### 3. Private-Clue Escape Room
- Concept: Two or more agents hold different clues and must selectively share or verify information to solve a collaborative puzzle.
- Why it fits: Success depends on modeling what others know and deciding what to reveal.
- Observation / actions: Private clue cards, shared room state, conversation history; actions are reveal clue, query teammate, manipulate object, or propose final code.
- Episode / termination: Ends when puzzle solved, wrong-code limit reached, or turn cap hits.
- Reward / verifier: Solve reward, partial reward for unlocking sub-puzzles, penalties for redundant disclosures and wrong final attempts.
- Anti-hacking safeguards: Server-held puzzle state, hidden clue templates, randomized clue wording, and action-cost penalty for spam.
- Why judges may care: Very intuitive and easy to watch.
- Demo story: Before training, the agent blurts everything or withholds useful clues; after training, it asks targeted questions and shares only necessary information.
- Build difficulty: `Low-Medium`
- Training feasibility: `High`; puzzle stages create good intermediate signals.
- Recommended stack: OpenEnv text/puzzle env; one trainable agent paired with scripted partner types; `GRPO`; no heavy SFT beyond action schema.
- 1-day prototype path: Build one puzzle family with randomized clue assignments and `4` actions.
- Biggest risk: Judges may see it as too close to a puzzle benchmark unless the clue-sharing mechanics feel novel.

### 4. Procurement Market Simulator
- Concept: Buyer, supplier, and compliance agents negotiate contracts under price, delivery, risk, and policy constraints.
- Why it fits: It captures realistic multi-agent incentives, deception pressure, and coalition behavior between internal stakeholders.
- Observation / actions: Demand forecasts, quotes, supplier reliability estimates, compliance flags, and chat history; actions are counteroffer, escalate, bundle, blacklist, or sign.
- Episode / termination: `6-12` rounds ending in signed contract or failed procurement.
- Reward / verifier: Utility from cost-quality-delivery mix, policy adherence, and penalties for risky or invalid contracts.
- Anti-hacking safeguards: Server-side contract checker, randomized supplier failure modes, delayed delivery outcomes, and hidden compliance rules.
- Why judges may care: Feels like a real business workflow rather than a toy game.
- Demo story: Trained agent learns to trade speed for reliability only when deadline pressure justifies it.
- Build difficulty: `Medium`
- Training feasibility: `Medium-High`
- Recommended stack: OpenEnv with structured quote objects; `GRPO`; likely light SFT for schema and negotiation style.
- 1-day prototype path: One buyer, `2` suppliers, `1` compliance rubric, and fixed delivery simulation.
- Biggest risk: If compliance logic is too simple, the game becomes only price minimization.

### 5. Startup Alliance Game
- Concept: Startup agents form partnerships, merge, or compete for market segments while investor pressure changes incentives.
- Why it fits: It targets coalition formation and strategic reasoning over multiple rounds.
- Observation / actions: Private runway, product strength, investor notes, public market trends, and alliance history; actions include pitch, partner, merge, poach, invest, or defend market.
- Episode / termination: Multi-round market simulation ending on valuation or survival horizon.
- Reward / verifier: Final firm value, survival, partnership success, and penalties for unstable coalitions.
- Anti-hacking safeguards: Hidden market shocks, delayed payoff realization, and server-controlled valuation model.
- Why judges may care: Strong storytelling and visually rich.
- Demo story: After training, the agent stops chasing flashy but value-destructive mergers.
- Build difficulty: `High`
- Training feasibility: `Medium-Low`; delayed rewards and many confounders.
- Recommended stack: OpenEnv with simplified market simulator; start with short horizons; `GRPO`; light curriculum mandatory.
- 1-day prototype path: Reduce to `3` firms, `2` market segments, and one merger mechanic.
- Biggest risk: Too abstract unless the economic model is tight and easy to explain.

### Theme 1 ranking

| Idea | Innovation | Story | Trainability | Reward | Risk | Fit | Weighted |
|---|---:|---:|---:|---:|---|---:|---:|
| GPU Budget Negotiation Arena | 9 | 9 | 9 | 9 | Medium | 10 | 9.0 |
| Disaster Relief Coalition Board | 8 | 10 | 7 | 8 | Med-High | 8 | 8.4 |
| Private-Clue Escape Room | 7 | 8 | 9 | 8 | Low-Med | 8 | 7.8 |
| Procurement Market Simulator | 8 | 8 | 8 | 9 | Medium | 9 | 8.2 |
| Startup Alliance Game | 9 | 9 | 5 | 7 | High | 6 | 7.8 |

- Best idea for winning: `GPU Budget Negotiation Arena`
- Best idea for fastest execution: `Private-Clue Escape Room`
- Best idea for strongest measurable reward improvement: `GPU Budget Negotiation Arena`

## Theme 2: (Super) Long-Horizon Planning & Instruction Following

### 1. 300-Instruction Office Ops Maze
- Concept: The agent receives hundreds of scattered instructions across docs, inboxes, and task boards, many of which interact or conflict.
- Why it fits: It tests long-horizon state tracking beyond a single prompt window and deliberate recovery from early mistakes.
- Observation / actions: Search tools over folders/messages, local scratchpad, action history, and partial task state; actions include open, search, extract, execute, defer, and summarize.
- Episode / termination: Long sessions with `50-150` steps; ends on task completion or budget exhaustion.
- Reward / verifier: Reward from completed instructions, conflict avoidance, order constraints, and penalties for skipped dependencies or repeated actions.
- Anti-hacking safeguards: Hidden dependency graph, server-side instruction state, anti-loop penalties, and protected scratchpad size limits.
- Why judges may care: Extremely aligned with the theme and easy to narrate as “instruction following beyond context memory.”
- Demo story: Baseline forgets earlier constraints; trained agent uses environment state effectively and finishes more tasks consistently.
- Build difficulty: `Medium-High`
- Training feasibility: `Medium`; needs shaped rewards and a constrained v1.
- Recommended stack: OpenEnv with file-board simulator; `GRPO` plus dense subtask rewards; light SFT for action grammar.
- 1-day prototype path: Reduce to `30-50` instructions with one search surface and dependency tags.
- Biggest risk: If the instruction graph is too large too early, reward becomes too sparse.

### 2. Codebase Refactor Quest
- Concept: The agent explores a toy multi-file repo, makes planned edits through tools, and must preserve tests while performing broader refactors.
- Why it fits: It is a long-horizon environment where mistakes compound and recovery matters.
- Observation / actions: File tree, grep/search outputs, diff summaries, failing tests, and edit budget; actions include inspect, edit, run targeted tests, revert own patch, and mark milestone.
- Episode / termination: Ends when all tests pass and refactor targets are satisfied, or edit/test budget is exhausted.
- Reward / verifier: Tests passing, required symbol moves, lint checks, minimal regressions, and penalties for thrashing edits.
- Anti-hacking safeguards: Protected files, hidden holdout tests, patch-size penalties, and restricted edit surface.
- Why judges may care: High practical value and easy before/after evaluation.
- Demo story: Untrained model over-edits and breaks tests; trained model sequences search, narrow edits, and verification.
- Build difficulty: `Medium`
- Training feasibility: `Medium-High`; strong verifiers but environment implementation must stay small.
- Recommended stack: OpenEnv repo sandbox; `GRPO`; light SFT on tool format; start with synthetic repos.
- 1-day prototype path: One `5-8` file codebase, `3` tests, one refactor target, and one hidden regression test.
- Biggest risk: If code-edit tools are too free-form, trajectories become expensive and unstable.

### 3. Research Planner Simulator
- Concept: The agent must plan literature review, choose experiments, spend budget, and revise strategy based on results over many turns.
- Why it fits: It stresses durable planning, memory, and adaptation after partial failure.
- Observation / actions: Paper cards, prior experiment outcomes, budget/time left, and hypothesis tracker; actions include read, propose hypothesis, run experiment, prune branch, or revise plan.
- Episode / termination: `20-40` decisions ending on final report quality and scientific score.
- Reward / verifier: Reward for discovering high-value findings efficiently, penalizing redundant experiments and unsupported claims.
- Anti-hacking safeguards: Hidden true mechanism, noisy experiment outcomes, cost accounting, and report-grounding checks.
- Why judges may care: Strong alignment to agentic research workflows.
- Demo story: Trained agent stops brute-forcing all experiments and learns adaptive experimentation.
- Build difficulty: `Medium`
- Training feasibility: `Medium`
- Recommended stack: OpenEnv symbolic science env; `GRPO`; likely no SFT beyond structured action outputs.
- 1-day prototype path: One hidden causal graph, `6` papers, `5` possible experiments, and a final report rubric.
- Biggest risk: If paper/experiment semantics are too abstract, the environment may feel synthetic.

### 4. Incident Response Runbook World
- Concept: The agent handles a multi-stage outage across alerts, logs, tickets, and customer impact while following a long runbook.
- Why it fits: It forces sequential diagnosis, action ordering, and recovery from wrong early branches.
- Observation / actions: Alerts, log snippets, ticket status, service graph, and runbook fragments; actions include inspect, restart, rollback, escalate, notify, and confirm recovery.
- Episode / termination: Ends on service recovery, SLA breach, or exhausted action budget.
- Reward / verifier: Service restoration, time-to-recovery, correct root-cause path, and penalties for unnecessary or harmful actions.
- Anti-hacking safeguards: Delayed effects, hidden root cause, action cooldowns, and server-side service-state simulator.
- Why judges may care: Feels real, dynamic, and not like a benchmark clone.
- Demo story: Trained agent learns staged triage instead of random restarts.
- Build difficulty: `Medium`
- Training feasibility: `High`; action space is finite and rewards are clear.
- Recommended stack: OpenEnv workflow env; `GRPO`; optional SFT for action schema only.
- 1-day prototype path: `3` services, `2` failure modes, `6` actions, and customer-impact meter.
- Biggest risk: Needs careful incident design so there is genuine planning rather than memorized action scripts.

### 5. Logistics Disruption Planner
- Concept: The agent manages deliveries across multiple days while weather, road closures, and inventory shocks appear late.
- Why it fits: It is a classic sparse-reward long-horizon planning problem with recoverable mistakes.
- Observation / actions: Map state, inventory, order deadlines, fleet status, and forecasts; actions are route, reroute, hold stock, prioritize orders, or request emergency shipment.
- Episode / termination: Multi-day horizon ending on fulfilled orders and cost metrics.
- Reward / verifier: On-time delivery, cost efficiency, spoilage avoidance, and penalties for deadhead miles and missed SLAs.
- Anti-hacking safeguards: Hidden shocks, conservation checks, and penalties for unrealistic reallocations.
- Why judges may care: Clear planning challenge with measurable outcomes.
- Demo story: Trained agent proactively buffers high-risk inventory and adapts to closures.
- Build difficulty: `Medium-High`
- Training feasibility: `Medium`
- Recommended stack: OpenEnv stateful simulator; `GRPO` with dense operational rewards; no SFT required.
- 1-day prototype path: `4` cities, `2` vehicle types, rolling weather shock, and `10` orders.
- Biggest risk: Can drift toward an operations-research simulator unless the LLM decision problem stays central.

### Theme 2 ranking

| Idea | Innovation | Story | Trainability | Reward | Risk | Fit | Weighted |
|---|---:|---:|---:|---:|---|---:|---:|
| 300-Instruction Office Ops Maze | 9 | 8 | 7 | 8 | Med-High | 8 | 8.3 |
| Codebase Refactor Quest | 8 | 8 | 8 | 9 | Medium | 9 | 8.1 |
| Research Planner Simulator | 8 | 8 | 7 | 8 | Medium | 8 | 7.8 |
| Incident Response Runbook World | 8 | 9 | 9 | 9 | Medium | 10 | 8.5 |
| Logistics Disruption Planner | 7 | 8 | 7 | 8 | Med-High | 7 | 7.4 |

- Best idea for winning: `Incident Response Runbook World`
- Best idea for fastest execution: `Codebase Refactor Quest`
- Best idea for strongest measurable reward improvement: `Incident Response Runbook World`

## Theme 3.1: World Modeling, Professional Tasks

### 1. API Incident Commander
- Concept: The agent operates across logs, metrics, deploy history, ticketing, and runbooks to resolve enterprise incidents in a partially observable system.
- Why it fits: It requires tool use, belief updates, and consistent world modeling across multiple systems.
- Observation / actions: Query logs, fetch metrics, inspect deploy diffs, change feature flags, escalate, and update tickets.
- Episode / termination: Ends on service stabilization, SLA breach, or exhausted ops budget.
- Reward / verifier: Recovery speed, correct fix, minimal collateral damage, and good ticketing hygiene.
- Anti-hacking safeguards: Hidden service dependencies, delayed effect of fixes, server-side state transitions, and no direct ground-truth root cause access.
- Why judges may care: Serious professional workflow with very clear evidence of learned behavior.
- Demo story: Trained policy learns targeted diagnosis before remediation.
- Build difficulty: `Medium`
- Training feasibility: `High`
- Recommended stack: OpenEnv tool ecosystem; `GRPO`; little or no SFT beyond tool-call formatting.
- 1-day prototype path: Simulate `3` services, deploy history, alerts, and a ticketing panel.
- Biggest risk: Overlap with Theme 2 if not framed around tool-centric world modeling.

### 2. Scientific Workflow Lab
- Concept: The agent goes from papers to code to experiments to interpretation in a simplified but stateful research loop.
- Why it fits: It tests causal reasoning, tool orchestration, and persistent internal state.
- Observation / actions: Read paper snippets, inspect notebook outputs, change parameters, launch experiments, compare metrics, and write conclusions.
- Episode / termination: Stops when budget ends or a valid discovery report is submitted.
- Reward / verifier: Reward for finding the hidden best hypothesis, efficient experiment design, and report correctness.
- Anti-hacking safeguards: Cost per experiment, noisy observations, hidden holdout benchmark, and report-grounding checks.
- Why judges may care: High ambition and strong research flavor.
- Demo story: Agent improves from random experiment spam to hypothesis-driven iteration.
- Build difficulty: `Medium-High`
- Training feasibility: `Medium`
- Recommended stack: OpenEnv research world; `GRPO`; likely light SFT for report schema.
- 1-day prototype path: One hidden function family, one dataset card view, and `5` experiment actions.
- Biggest risk: Hard to make believable without too much custom content.

### 3. Tool Discovery Benchmark
- Concept: The environment contains a large toolbox with sparse documentation, and the agent must discover the right tools and argument combinations to solve tasks.
- Why it fits: It directly targets world modeling over tool capabilities and partial observability.
- Observation / actions: Tool docs, prior outputs, task brief, and memory; actions are inspect tool, call tool, chain outputs, and submit result.
- Episode / termination: Ends on task success or call-budget exhaustion.
- Reward / verifier: Final task correctness, tool-call efficiency, and penalties for invalid or redundant calls.
- Anti-hacking safeguards: Hidden tool quirks, rate limits, invalid-parameter penalties, and distinct train/holdout tool compositions.
- Why judges may care: Broad relevance to real agent systems.
- Demo story: Baseline flails through tools; trained agent learns exploratory then exploitative tool usage.
- Build difficulty: `Low-Medium`
- Training feasibility: `High`
- Recommended stack: OpenEnv MCP-like tool world; `GRPO`; no SFT necessary if actions are structured.
- 1-day prototype path: `8-10` tools, `3` task templates, and a docs browser.
- Biggest risk: If the tools are too toy-like, judges may see it as a planner benchmark rather than a world model.

### 4. Browser CRM Renewal Desk
- Concept: The agent navigates a CRM-style browser app, billing panel, and policy docs to rescue at-risk customer renewals.
- Why it fits: It is a professional browser workflow with dynamic state and multiple tools.
- Observation / actions: Structured browser observations, account history, support notes, policy lookup, and action buttons.
- Episode / termination: Ends on renewal success, churn, or compliance failure.
- Reward / verifier: Renewal outcome, discount efficiency, policy compliance, and customer sentiment stability.
- Anti-hacking safeguards: Hidden customer preferences, anti-shortcut state checks, and policy-based action blockers.
- Why judges may care: Looks polished in demo and is grounded in real enterprise work.
- Demo story: The trained agent learns when to offer discounts, escalate, or solve issues first.
- Build difficulty: `Medium`
- Training feasibility: `Medium`
- Recommended stack: OpenEnv browser/task env; `GRPO`; light SFT may help with action serialization.
- 1-day prototype path: One simple web app with account panel, policy tab, and `3` retention levers.
- Biggest risk: Browser complexity can eat hackathon time.

### 5. Finance Reconciliation Desk
- Concept: The agent reconciles invoices, payment records, approvals, and vendor emails across noisy enterprise systems.
- Why it fits: It requires cross-tool state tracking, belief revision, and consistent action under incomplete information.
- Observation / actions: ERP rows, invoices, emails, approval logs; actions are match, request clarification, escalate, approve, or hold.
- Episode / termination: Ends when batch is closed or too many erroneous approvals happen.
- Reward / verifier: Correct reconciliation, low false approvals, latency, and audit completeness.
- Anti-hacking safeguards: Hidden fraud patterns, holdout edge cases, and penalties for unsafe auto-approval.
- Why judges may care: Concrete, business-relevant, and highly verifiable.
- Demo story: Trained policy learns to ask for clarification only when ambiguity is real.
- Build difficulty: `Low-Medium`
- Training feasibility: `High`
- Recommended stack: OpenEnv tabular/email hybrid env; `GRPO`; no SFT needed besides schema.
- 1-day prototype path: One batch of `20` invoices, `3` email patterns, and `2` fraud edge cases.
- Biggest risk: Less visually exciting than browser-heavy alternatives.

### Theme 3.1 ranking

| Idea | Innovation | Story | Trainability | Reward | Risk | Fit | Weighted |
|---|---:|---:|---:|---:|---|---:|---:|
| API Incident Commander | 8 | 9 | 9 | 9 | Medium | 10 | 8.6 |
| Scientific Workflow Lab | 9 | 8 | 6 | 8 | Med-High | 7 | 8.0 |
| Tool Discovery Benchmark | 8 | 7 | 9 | 8 | Low-Med | 9 | 7.9 |
| Browser CRM Renewal Desk | 8 | 9 | 7 | 8 | Medium | 8 | 8.1 |
| Finance Reconciliation Desk | 7 | 7 | 9 | 9 | Low-Med | 9 | 7.6 |

- Best idea for winning: `API Incident Commander`
- Best idea for fastest execution: `Finance Reconciliation Desk`
- Best idea for strongest measurable reward improvement: `API Incident Commander`

## Theme 3.2: World Modeling, Personalized Tasks

### 1. Executive Calendar Conflict Resolver
- Concept: The agent manages meetings, travel, dinner plans, pickups, and personal constraints for a simulated executive with shifting priorities.
- Why it fits: It captures realistic personal delegation and conflict resolution with hidden preferences.
- Observation / actions: Calendar, inbox, contacts, preferences, location/travel estimates, and family constraints; actions include move, cancel, delegate, draft reply, book slot, and ask follow-up.
- Episode / termination: Ends when the day/week plan stabilizes or too many conflicts remain unresolved.
- Reward / verifier: Constraint satisfaction, preference alignment, relationship preservation, travel feasibility, and low churn in schedule changes.
- Anti-hacking safeguards: Hidden preference weights, server-side travel/time checks, and penalties for fake confirmations or impossible schedules.
- Why judges may care: Extremely relatable, visually strong, and easy to explain to non-technical reviewers.
- Demo story: Baseline double-books dinner and meetings; trained agent preserves critical obligations while minimizing social fallout.
- Build difficulty: `Medium`
- Training feasibility: `High`; rewards can be decomposed well.
- Recommended stack: OpenEnv personal-assistant simulator; `GRPO`; light SFT for email/calendar action schema.
- 1-day prototype path: One day, `10` events, `5` contacts, commute graph, and preference rubric.
- Biggest risk: Needs clear reward shaping so it is more than “calendar packing.”

### 2. Tough Email Reply and Delegation Desk
- Concept: The agent drafts responses to difficult personal and professional emails while deciding when to defer, soften tone, or delegate.
- Why it fits: It models personal assistant work with hidden relationship stakes.
- Observation / actions: Thread history, sender profile, past relationship state, pending tasks; actions are draft, choose tone, ask clarifying question, delegate, or defer.
- Episode / termination: Ends when inbox is cleared or relationship score falls below threshold.
- Reward / verifier: Correct intent fulfillment, tone fit, constraint compliance, and downstream task completion.
- Anti-hacking safeguards: Hidden sender preferences, no reward for empty politeness, structured outcome checks on delegated tasks.
- Why judges may care: Very intuitive and personally relevant.
- Demo story: Trained agent learns to separate apology, clarification, and action commitment instead of sending generic safe replies.
- Build difficulty: `Low-Medium`
- Training feasibility: `Medium`; some tone scoring may tempt LLM-as-judge, so hard backstops are needed.
- Recommended stack: OpenEnv messaging env; hybrid reward with hard task checks plus rubric score; light SFT likely helpful.
- 1-day prototype path: `15` thread templates, `4` relationship types, and explicit action slots in the reply object.
- Biggest risk: Over-reliance on subjective judging if not grounded in structured outcomes.

### 3. Dinner and Drive Crisis Planner
- Concept: The agent must reconcile work overruns, pickups, reservations, traffic, and family preferences to salvage the evening.
- Why it fits: It is a realistic personal planning environment with dynamic conflicts and delegation.
- Observation / actions: Traffic map, calendars, reservation windows, family messages, and location state; actions include reroute, reschedule, delegate pickup, notify contacts, and book alternative.
- Episode / termination: Ends when the evening concludes or key commitments fail.
- Reward / verifier: On-time arrival, priority satisfaction, budget, and minimized disappointment score.
- Anti-hacking safeguards: Hidden traffic shocks, travel validation, reservation lockouts, and explicit dependency checks.
- Why judges may care: Strong, relatable story and compact scope.
- Demo story: Baseline reacts too late; trained agent proactively reroutes and communicates.
- Build difficulty: `Low-Medium`
- Training feasibility: `High`
- Recommended stack: OpenEnv event planner; `GRPO`; no major SFT needed beyond action structure.
- 1-day prototype path: One evening, `3` commitments, `2` delegates, `1` traffic disruption generator.
- Biggest risk: Scope may feel too small unless framed as a family-conflict micro-world.

### 4. Household Budget and Shopping Mediator
- Concept: The agent plans grocery/household purchases while balancing dietary needs, budget, stock levels, and family preferences.
- Why it fits: It is a personalized delegation task with persistent state and conflict resolution.
- Observation / actions: Pantry, budget, shopping options, dietary rules, and family preference profiles; actions are purchase, substitute, defer, ask, or bundle.
- Episode / termination: Week ends or budget/stock crises trigger failure.
- Reward / verifier: Budget adherence, dietary compliance, satisfaction score, and stockout avoidance.
- Anti-hacking safeguards: Hidden preference weights, inventory accounting, and penalties for impossible substitutions.
- Why judges may care: Personalized but still objectively checkable.
- Demo story: Trained agent learns strategic substitution instead of naive cheapest-item behavior.
- Build difficulty: `Low`
- Training feasibility: `High`
- Recommended stack: OpenEnv shopping simulator; `GRPO`; no SFT needed.
- 1-day prototype path: One family, pantry state, `30` purchasable items, and `5` recurring needs.
- Biggest risk: Lower “wow” factor than schedule or messaging tasks.

### 5. Family Weekend Negotiation Planner
- Concept: The agent resolves competing family requests, weather constraints, travel limits, and emotional priorities to build a weekend plan.
- Why it fits: It combines negotiation, preferences, and personalized scheduling in a partially observable world.
- Observation / actions: Preference cards, weather, travel times, fixed commitments, and conversation state; actions are propose plan, negotiate swap, book, cancel, or split time.
- Episode / termination: Ends when weekend plan is finalized or too many members disengage.
- Reward / verifier: Constraint satisfaction, fairness, budget, and family satisfaction score.
- Anti-hacking safeguards: Hidden preference intensity, booking deadlines, and penalties for contradictory promises.
- Why judges may care: High storytelling value and easy to visualize.
- Demo story: Trained agent learns fairness and trade-offs instead of over-serving one family member.
- Build difficulty: `Medium`
- Training feasibility: `Medium-High`
- Recommended stack: OpenEnv planner/negotiation env; `GRPO`; possible light SFT for communication acts.
- 1-day prototype path: `4` family members, `6` activities, weather generator, and budget.
- Biggest risk: Could drift toward a scheduling toy if preference interactions are weak.

### Theme 3.2 ranking

| Idea | Innovation | Story | Trainability | Reward | Risk | Fit | Weighted |
|---|---:|---:|---:|---:|---|---:|---:|
| Executive Calendar Conflict Resolver | 8 | 10 | 8 | 8 | Medium | 10 | 8.6 |
| Tough Email Reply and Delegation Desk | 7 | 9 | 6 | 6 | Medium | 7 | 7.3 |
| Dinner and Drive Crisis Planner | 7 | 9 | 8 | 8 | Low-Med | 9 | 7.9 |
| Household Budget and Shopping Mediator | 6 | 7 | 9 | 9 | Low | 8 | 7.1 |
| Family Weekend Negotiation Planner | 8 | 9 | 8 | 8 | Medium | 9 | 8.3 |

- Best idea for winning: `Executive Calendar Conflict Resolver`
- Best idea for fastest execution: `Dinner and Drive Crisis Planner`
- Best idea for strongest measurable reward improvement: `Executive Calendar Conflict Resolver`

## Theme 4: Self-Improvement

### 1. Negotiation Self-Play Ladder
- Concept: Agents repeatedly negotiate resource deals against cloned or curriculum opponents that adapt to exploit previous weaknesses.
- Why it fits: It naturally supports self-play, difficulty escalation, and recursive capability growth.
- Observation / actions: Same as negotiation env, plus opponent archetype summaries and prior exploit traces.
- Episode / termination: Short negotiation episodes rolled into leagues/ladders.
- Reward / verifier: Negotiation utility, robustness against exploit strategies, and bonus for beating stronger ladder tiers.
- Anti-hacking safeguards: Hidden opponent seeds, rotating archetypes, and holdout opponent sets.
- Why judges may care: Clear self-improvement story with visible curriculum.
- Demo story: Agent first beats naive opponents, then learns to handle deceptive ones introduced by the ladder.
- Build difficulty: `Medium`
- Training feasibility: `High`
- Recommended stack: Start from Theme 1 GPU negotiation core; add population/self-play scheduler; `GRPO`.
- 1-day prototype path: Self-play over `3` scripted opponent tiers with periodic policy snapshot evaluation.
- Biggest risk: Needs careful evaluation so improvement is not just overfitting to one opponent pool.

### 2. Auto-Generated Refactor Arena
- Concept: The environment procedurally generates small codebases and refactor goals whose difficulty increases when the agent succeeds.
- Why it fits: It is a clean adaptive curriculum environment rather than a static benchmark.
- Observation / actions: File tree, tests, edit tools, goal spec, and failure traces.
- Episode / termination: Ends on passing hidden/public tests or budget exhaustion.
- Reward / verifier: Same as code task, plus curriculum score for succeeding on harder generated instances.
- Anti-hacking safeguards: Template randomization, hidden tests, protected files, and difficulty held-out generators.
- Why judges may care: Strong “self-improving curriculum” narrative.
- Demo story: Agent advances from single-file renames to multi-file signature migrations.
- Build difficulty: `Medium-High`
- Training feasibility: `Medium`
- Recommended stack: OpenEnv synthetic repo generator; `GRPO`; light SFT for tool syntax.
- 1-day prototype path: Two code templates, variable/function rename goals, and test generator.
- Biggest risk: Generator quality determines everything; weak templates feel repetitive.

### 3. Tasksmith Puzzle Forge
- Concept: One model generates logic or tool-use tasks specifically designed to expose the solver’s current blind spots.
- Why it fits: It is explicit recursive skill amplification via adversarial task generation.
- Observation / actions: Generator sees solver stats and recent failures; solver interacts with generated tasks.
- Episode / termination: Dual loop of task generation then solve attempt.
- Reward / verifier: Generator rewarded for tasks just above solver ability; solver rewarded for solving increasingly hard tasks.
- Anti-hacking safeguards: Difficulty validators, novelty checks, and holdout evaluator to prevent generator producing degenerate impossible tasks.
- Why judges may care: Very ambitious and research-flavored.
- Demo story: The system learns to manufacture training signal rather than consume a static dataset.
- Build difficulty: `High`
- Training feasibility: `Medium-Low`; more moving pieces than a typical hackathon can absorb.
- Recommended stack: Two-policy or alternating-role OpenEnv env; `GRPO`; start with symbolic puzzle templates only.
- 1-day prototype path: Generator chooses puzzle parameters from a bounded family rather than free-form task creation.
- Biggest risk: Hard to stabilize and hard to explain cleanly in a short demo.

### 4. Adaptive Tool-Discovery Curriculum
- Concept: The environment monitors which tool confusions the agent still has and procedurally serves harder compositions of those tools.
- Why it fits: It is self-improvement via targeted curriculum rather than fixed tasks.
- Observation / actions: Tool docs, failure history, current curriculum level, and task brief.
- Episode / termination: Standard task episodes with dynamic next-task selection.
- Reward / verifier: Task success plus extra learning progress signal on newly mastered tool compositions.
- Anti-hacking safeguards: Separate train/test tool mixes, curriculum regularization, and anti-shortcut penalties for brute-force tool calling.
- Why judges may care: Practical, believable, and visibly adaptive.
- Demo story: The curriculum discovers the agent fails on auth+pagination workflows, then keeps serving those until performance improves.
- Build difficulty: `Medium`
- Training feasibility: `High`
- Recommended stack: Extend Theme 3 tool-discovery env with adaptive sampler; `GRPO`.
- 1-day prototype path: Tag tasks by skill and make scheduler pick underperforming tags.
- Biggest risk: The self-improvement component may feel like a sampler heuristic unless shown clearly.

### 5. Adversarial Scheduler Arena
- Concept: The environment escalates scheduling conflicts and hidden constraints based on the agent’s current success frontier.
- Why it fits: It adapts difficulty automatically and can train richer planning behavior through self-play-like curriculum.
- Observation / actions: Calendar/task state plus generated conflict scenarios.
- Episode / termination: Day/week planning episodes with increasing complexity.
- Reward / verifier: Constraint satisfaction, preference preservation, and harder-scenario success bonus.
- Anti-hacking safeguards: Difficulty validator, hidden constraints, and held-out scenario families.
- Why judges may care: Combines personal-assistant usefulness with adaptive training.
- Demo story: Model graduates from simple conflicts to dense multi-stakeholder weeks.
- Build difficulty: `Low-Medium`
- Training feasibility: `High`
- Recommended stack: Extend Theme 3.2 calendar env with procedural scenario generator; `GRPO`.
- 1-day prototype path: Scenario difficulty ladder driven by event count and hidden dependencies.
- Biggest risk: Less pure self-play than the theme examples, so the framing has to be crisp.

### Theme 4 ranking

| Idea | Innovation | Story | Trainability | Reward | Risk | Fit | Weighted |
|---|---:|---:|---:|---:|---|---:|---:|
| Negotiation Self-Play Ladder | 9 | 9 | 8 | 8 | Medium | 10 | 8.7 |
| Auto-Generated Refactor Arena | 9 | 8 | 7 | 9 | Med-High | 8 | 8.4 |
| Tasksmith Puzzle Forge | 10 | 8 | 5 | 7 | High | 6 | 8.1 |
| Adaptive Tool-Discovery Curriculum | 8 | 8 | 9 | 9 | Medium | 9 | 8.2 |
| Adversarial Scheduler Arena | 7 | 8 | 9 | 8 | Low-Med | 8 | 7.8 |

- Best idea for winning: `Negotiation Self-Play Ladder`
- Best idea for fastest execution: `Adversarial Scheduler Arena`
- Best idea for strongest measurable reward improvement: `Adaptive Tool-Discovery Curriculum`

## Theme 5: Wild Card

### 1. Open Source Maintainer Arena
- Concept: The agent triages issues, reviews patches, replies to contributors, and decides what to merge in a simulated open-source project.
- Why it is compelling: It is a realistic, underexplored agent environment with visible long-term project health.
- Observation / actions: Issue tracker, PR diffs, CI status, contributor history, and roadmap; actions include label, comment, request changes, merge, close, or defer.
- Episode / termination: Sprint-style episode ending on project-health score.
- Reward / verifier: Bug resolution, CI health, community satisfaction, and penalty for regressions or toxic interactions.
- Anti-hacking safeguards: Hidden regression tests, contributor reliability scores, and no reward for superficial issue closure.
- Why judges may care: Highly relatable for developers and clearly useful.
- Demo story: Trained agent stops merging flashy but unstable PRs and learns better triage.
- Build difficulty: `Medium`
- Training feasibility: `Medium-High`
- Recommended stack: OpenEnv issue/PR simulator; `GRPO`; light SFT for response schema.
- 1-day prototype path: `10` issues, `5` PRs, CI checks, and contributor personas.
- Biggest risk: Needs tight simulation logic so it is more than a queue sorter.

### 2. Fraud Investigation War Room
- Concept: The agent investigates suspicious transactions, requests evidence, freezes accounts, and balances fraud prevention against false positives.
- Why it is compelling: It is high stakes, dynamic, and strongly verifiable.
- Observation / actions: Transaction graph, account metadata, alert feed, and evidence requests; actions are inspect, freeze, escalate, clear, or request more data.
- Episode / termination: Ends when the case batch closes or too many bad actions are taken.
- Reward / verifier: Fraud caught, false-positive rate, investigation cost, and audit completeness.
- Anti-hacking safeguards: Hidden fraud ring patterns, holdout attack strategies, and penalties for mass freezing.
- Why judges may care: Novel, practical, and easy to measure.
- Demo story: Trained agent learns selective escalation instead of brute-force freezes.
- Build difficulty: `Medium`
- Training feasibility: `High`
- Recommended stack: OpenEnv graph/case env; `GRPO`; no major SFT needed.
- 1-day prototype path: Small transaction graph, `3` fraud templates, and an evidence-query action.
- Biggest risk: Visualization may take extra work if you want the graph story to land well.

### 3. Museum Memory Heist
- Concept: The agent reconstructs a theft from witness reports, camera clips, access logs, and room layouts while false leads are planted.
- Why it is compelling: It combines memory, reasoning, and investigation into a distinctive narrative.
- Observation / actions: Evidence board, room map, witness statements, timeline tools; actions are inspect, cross-check, infer, request footage, or accuse.
- Episode / termination: Ends on accusation or evidence budget exhaustion.
- Reward / verifier: Correct culprit and timeline reconstruction, evidence efficiency, and penalties for false accusations.
- Anti-hacking safeguards: Randomized culprit generation, contradictory witness noise, and hidden holdout clues.
- Why judges may care: Very strong storytelling and visual demo.
- Demo story: Agent evolves from shallow clue matching to coherent timeline-building.
- Build difficulty: `Medium`
- Training feasibility: `Medium`
- Recommended stack: OpenEnv investigation env; `GRPO`; no SFT beyond action schema.
- 1-day prototype path: One museum layout, `4` suspects, `8` evidence items, and timeline checker.
- Biggest risk: Needs a crisp verifier so it is not judged as pure mystery storytelling.

### 4. Startup CEO Simulator
- Concept: The agent runs a startup across hiring, pricing, shipping, support, and fundraising under stochastic market pressure.
- Why it is compelling: It is broad, ambitious, and can mix planning, negotiation, and world modeling.
- Observation / actions: KPIs, team state, burn, product backlog, investor updates; actions are hire, fire, ship, price, fundraise, or pivot.
- Episode / termination: Multi-quarter company simulation ending on valuation or shutdown.
- Reward / verifier: Survival, growth, runway, product quality, and employee churn penalties.
- Anti-hacking safeguards: Hidden market shocks, delayed action effects, and server-side economy model.
- Why judges may care: Memorable and unusual.
- Demo story: Trained agent learns disciplined growth instead of vanity metrics.
- Build difficulty: `High`
- Training feasibility: `Medium-Low`
- Recommended stack: OpenEnv business simulator; `GRPO`; mandatory curriculum.
- 1-day prototype path: Reduce to `3` levers: pricing, hiring, and marketing.
- Biggest risk: Easy to overscope and hard to prove training signal quickly.

### 5. Contract Redline Negotiator
- Concept: The agent revises legal/commercial clauses against a counterparty under business and risk constraints.
- Why it is compelling: It is a crisp, document-grounded workflow with strategic negotiation flavor.
- Observation / actions: Clause library, redlines, playbook, risk tolerances, and counterparty stance; actions are accept, redline, counter, escalate, or justify.
- Episode / termination: Ends on signed contract or breakdown.
- Reward / verifier: Playbook compliance, issue resolution, business-value preservation, and cycle efficiency.
- Anti-hacking safeguards: Hidden risk weights, clause validator, and no reward for unsupported justifications.
- Why judges may care: Real professional utility and structured rewards.
- Demo story: Trained policy learns targeted edits instead of maximal redlining.
- Build difficulty: `Medium`
- Training feasibility: `Medium-High`
- Recommended stack: OpenEnv negotiation/doc env; `GRPO`; light SFT for structured clause edits.
- 1-day prototype path: `10` clauses, `3` counterparty archetypes, and a playbook checker.
- Biggest risk: Needs a simple contract language so judges can follow the outcome fast.

### Theme 5 ranking

| Idea | Innovation | Story | Trainability | Reward | Risk | Fit | Weighted |
|---|---:|---:|---:|---:|---|---:|---:|
| Open Source Maintainer Arena | 9 | 9 | 7 | 8 | Medium | 9 | 8.5 |
| Fraud Investigation War Room | 9 | 9 | 8 | 9 | Medium | 10 | 8.8 |
| Museum Memory Heist | 8 | 10 | 7 | 8 | Medium | 8 | 8.4 |
| Startup CEO Simulator | 10 | 9 | 5 | 7 | High | 6 | 8.1 |
| Contract Redline Negotiator | 8 | 8 | 8 | 9 | Medium | 8 | 8.1 |

- Best idea for winning: `Fraud Investigation War Room`
- Best idea for fastest execution: `Contract Redline Negotiator`
- Best idea for strongest measurable reward improvement: `Fraud Investigation War Room`

## Top picks across all themes

### Top 5 overall
1. `GPU Budget Negotiation Arena` - strongest balance of novelty, clear reward, short episodes, and theme purity.
2. `API Incident Commander` - very trainable, strong demo story, and excellent professional relevance.
3. `Executive Calendar Conflict Resolver` - best non-technical storytelling and high judge accessibility.
4. `Negotiation Self-Play Ladder` - strongest self-improvement story with visible training progression.
5. `Fraud Investigation War Room` - compelling wild-card option with sharp verification and high practical value.

### Top 3 safest submissions
1. `API Incident Commander`
2. `GPU Budget Negotiation Arena`
3. `Executive Calendar Conflict Resolver`

### Top 3 highest-upside / most ambitious
1. `Negotiation Self-Play Ladder`
2. `Scientific Workflow Lab`
3. `Startup CEO Simulator`

## Final recommendation

### Recommended overall pick: GPU Budget Negotiation Arena

Why this one:
- It is the cleanest fit for `Theme 1`
- The story is instantly understandable
- Rewards are objective and composable
- Training can show visible improvement fast
- It naturally supports a strong v2 extension into `Theme 4` via self-play ladders

### Environment scope
- `3-5` labs compete for `GPU` blocks over `5-10` rounds
- Each lab has hidden job queue, urgency, budget sensitivity, and utility curve
- Public state shows market availability and accepted contracts
- Actions:
  - `send_offer`
  - `accept_offer`
  - `reject_offer`
  - `counter_offer`
  - `reserve_capacity`
  - `form_coalition`
  - `commit_trade`
- Observation includes own jobs, visible market slots, message history, and coalition status

### V1 feature set
- Single trainable agent versus `2-3` scripted opponent archetypes:
  - selfish maximizer
  - deadline panicker
  - cooperative trader
- One resource type first: `GPU hours`
- Jobs vary on deadline, value, and minimum required compute
- Final utility = completed job value minus spend plus coalition bonus
- Conversation can be structured JSON plus one short free-text note for demo flavor

### Reward rubric
- `+1.0 to +5.0` for completing high-value jobs
- `+0.5 to +2.0` for Pareto-improving trades
- `-0.5 to -2.0` for idle reserved capacity
- `-1.0` for invalid or impossible offers
- `-1.5` for breaking a coalition commitment
- `-0.2` for repetitive spam offers
- Optional shaping:
  - positive reward for matching urgency to acquisition strategy
  - small penalty for overpaying relative to hidden fair-value band

### Anti-hacking controls
- Server-side contract execution only
- No access to hidden opponent utility
- Randomized job mixes and urgency profiles per episode
- Distinct holdout scenario seeds for evaluation
- Penalty for repeated degenerate offer loops
- Hard validation on resource conservation and contract feasibility

### Training plan
- Warm start:
  - tiny SFT on action schema and valid negotiation formatting
  - `50-200` synthetic expert traces are enough
- RL:
  - `GRPO` with short rollouts
  - batch scenarios sampled from easy-to-medium job distributions
  - start with `1` trainable agent and scripted opponents
- Evaluation:
  - reward over training
  - completed-job value
  - accepted-deal rate
  - exploit/invalid-offer rate
  - head-to-head versus baseline policy

### HF Space deployment path
- Build OpenEnv env locally
- Expose FastAPI app with `reset`, `step`, and `state`
- Push early to HF Space for remote testing
- Keep one simple web panel:
  - current jobs
  - incoming offers
  - chosen action
  - final score breakdown

### README / demo story
- Problem: LLMs are weak at strategic multi-agent reasoning under hidden incentives
- Environment: labs negotiate for scarce compute with incomplete information
- Reward: utility, trade quality, and commitment reliability
- Result: trained model secures better allocations than baseline
- Demo:
  - baseline negotiation clip
  - reward plot
  - trained negotiation clip
  - one example of coalition formation or smart concession

### Team split for 3-4 people
- Person A: OpenEnv environment, episode logic, HF Space deployment
- Person B: reward rubric, opponent policies, anti-cheat checks
- Person C: TRL/Unsloth training script, evaluation plots, checkpoint handling
- Person D: demo UI, README, mini-blog/video, scenario visualization

### Suggested 2-day execution order
1. Day 0 pre-onsite:
   - implement env skeleton
   - finish one end-to-end scripted episode
   - lock reward columns
   - deploy Space
2. Day 1 morning:
   - add scripted opponents
   - generate synthetic warm-start traces
   - run tiny baseline evaluations
3. Day 1 afternoon:
   - launch first GRPO run
   - inspect rollouts for spam, deadlocks, and invalid offers
   - tune shaping only if needed
4. Day 2 morning:
   - run best training config on more seeds
   - save plots and head-to-head evaluation
   - create before/after transcripts
5. Day 2 afternoon:
   - finalize README
   - record `<=2 min` demo
   - link Space, notebook, plots, and video from README

### If you want the absolute safest fallback
- Pick `API Incident Commander`

### If you want the most judge-friendly non-technical story
- Pick `Executive Calendar Conflict Resolver`

### If you want the strongest self-improvement narrative
- Pick `Negotiation Self-Play Ladder`
