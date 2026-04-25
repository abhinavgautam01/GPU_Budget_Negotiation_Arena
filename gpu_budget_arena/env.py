from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

from gpu_budget_arena.judge import RuleBasedJudge
from gpu_budget_arena.models import (
    ActionResult,
    CoalitionView,
    Difficulty,
    EnvironmentState,
    GpuBlockView,
    GpuNegotiationAction,
    GpuNegotiationObservation,
    JobView,
    LabState,
    MessageView,
    OfferView,
    PublicMarketView,
    ResetConfig,
    RewardBreakdown,
    TaskType,
    VisibleLabView,
)


class GpuBudgetNegotiationEnv:
    def __init__(self) -> None:
        self.state_data: EnvironmentState | None = None
        self.rng = random.Random(42)

    def reset(self, config: ResetConfig | dict[str, Any] | None = None) -> GpuNegotiationObservation:
        reset_config = config if isinstance(config, ResetConfig) else ResetConfig.model_validate(config or {})
        difficulty = reset_config.difficulty or self._difficulty_for_task(reset_config.task_type)
        self.rng = random.Random(reset_config.seed)
        self.state_data = self._generate_world(
            reset_config.task_type,
            difficulty,
            reset_config.seed,
            reset_config.judge_mode,
        )
        return self._observation()

    def step(self, action: GpuNegotiationAction | dict[str, Any]) -> GpuNegotiationObservation:
        state = self._require_state()
        if state.done:
            return self._observation(ActionResult(ok=False, code="episode_done", message="Episode is already done."))

        try:
            parsed = action if isinstance(action, GpuNegotiationAction) else GpuNegotiationAction.model_validate(action)
        except Exception as exc:  # pydantic errors are intentionally reported through the env channel.
            result = ActionResult(ok=False, code="malformed_action", message=str(exc))
            self._apply_reward(self._reward_for_result(result, invalid_penalty=-0.15), result)
            self._advance_after_action()
            return self._observation()

        result, deal_delta = self._apply_agent_action(parsed)
        reward = self._reward_for_result(result, deal_delta=deal_delta)
        self._apply_reward(reward, result)
        self._advance_after_action()
        return self._observation()

    def state(self) -> dict[str, Any]:
        state = self._require_state()
        return state.model_dump()

    def _require_state(self) -> EnvironmentState:
        if self.state_data is None:
            self.reset()
        assert self.state_data is not None
        return self.state_data

    def _difficulty_for_task(self, task_type: TaskType) -> Difficulty:
        return {
            "single_trade": "easy",
            "market_round": "medium",
            "coalition_market": "hard",
        }[task_type]

    def _generate_world(
        self,
        task_type: TaskType,
        difficulty: Difficulty,
        seed: int,
        judge_mode: str = "off",
    ) -> EnvironmentState:
        lab_count = {"easy": 2, "medium": 3, "hard": self.rng.choice([4, 5])}[difficulty]
        max_rounds = {"easy": 3, "medium": 5, "hard": 9}[difficulty]
        archetypes = ["cooperative", "selfish", "deadline_panicked", "deceptive", "retaliatory"]
        labs: dict[str, LabState] = {}
        blocks: dict[str, GpuBlockView] = {}

        for i in range(lab_count):
            lab_id = f"lab_{i}"
            lab = LabState(
                lab_id=lab_id,
                public_name="You" if i == 0 else f"Lab {chr(64 + i)}",
                budget=round(self.rng.uniform(70, 130), 2),
                reputation=0.65 if i == 0 else round(self.rng.uniform(0.35, 0.85), 2),
                opponent_archetype=None if i == 0 else archetypes[(i - 1) % len(archetypes)],
            )
            block_count = {"easy": 2, "medium": 3, "hard": 4}[difficulty]
            for j in range(block_count):
                block_id = f"b_{i}_{j}"
                block = GpuBlockView(
                    block_id=block_id,
                    start_round=self.rng.randint(0, max(1, max_rounds - 2)),
                    duration=self.rng.choice([1, 2, 3]),
                    gpu_count=self.rng.choice([1, 2, 4]),
                    reliability=round(self.rng.uniform(0.72, 0.99), 2),
                    energy_cost=round(self.rng.uniform(1.5, 6.0), 2),
                    owner_lab_id=lab_id,
                    status="available",
                )
                blocks[block_id] = block
                lab.owned_blocks.append(block_id)

            job_count = {"easy": 1, "medium": 2, "hard": 3}[difficulty]
            for j in range(job_count):
                required = float(self.rng.choice([2, 3, 4, 6, 8]))
                lab.private_jobs.append(
                    JobView(
                        job_id=f"j_{i}_{j}",
                        gpu_hours_required=required,
                        deadline_round=self.rng.randint(2, max_rounds),
                        base_value=round(self.rng.uniform(30, 95), 2),
                        urgency_multiplier=round(self.rng.uniform(1.0, 1.8), 2),
                        min_reliability=round(self.rng.uniform(0.70, 0.90), 2),
                        partial_credit_allowed=self.rng.random() < 0.35,
                        private_notes=self.rng.choice(["paper_deadline", "customer_incident", "benchmark_run"]),
                    )
                )
            labs[lab_id] = lab

        shock_schedule: dict[int, str] = {}
        if difficulty == "medium":
            shock_schedule[3] = "capacity_failure"
        elif difficulty == "hard":
            shock_schedule[3] = "capacity_failure"
            shock_schedule[6] = "energy_spike"
            shock_schedule[8] = "reliability_degradation" if seed % 2 == 0 else "demand_surge"

        state = EnvironmentState(
            task_id=f"{task_type}-{seed}",
            task_type=task_type,
            difficulty=difficulty,
            seed=seed,
            judge_mode=judge_mode,
            round_index=0,
            max_rounds=max_rounds,
            controlled_lab_id="lab_0",
            labs=labs,
            blocks=blocks,
            offers={},
            coalitions={},
            messages=[],
            shock_schedule=shock_schedule,
            shock_history=[],
        )
        self.state_data = state
        self._seed_initial_opportunities()
        return state

    def _seed_initial_opportunities(self) -> None:
        state = self._require_state()
        if state.difficulty == "easy":
            return
        for lab_id in list(state.labs.keys()):
            if lab_id == state.controlled_lab_id:
                continue
            own = state.labs[lab_id].owned_blocks[:1]
            requested = state.labs[state.controlled_lab_id].owned_blocks[:1]
            if own and requested:
                offer_id = self._new_offer_id()
                state.offers[offer_id] = OfferView(
                    offer_id=offer_id,
                    from_lab_id=lab_id,
                    to_lab_id=state.controlled_lab_id,
                    round_created=0,
                    expires_round=2,
                    offered_blocks=own,
                    requested_blocks=requested,
                    payment=round(self.rng.uniform(2, 8), 2),
                    message="Initial capacity swap proposal.",
                )

    def _apply_agent_action(self, action: GpuNegotiationAction) -> tuple[ActionResult, float]:
        handlers = {
            "send_offer": self._send_offer,
            "accept_offer": self._accept_offer,
            "reject_offer": self._reject_offer,
            "counter_offer": self._counter_offer,
            "make_pitch": self._make_pitch,
            "counter_pitch": self._make_pitch,
            "reserve_capacity": self._reserve_capacity,
            "release_capacity": self._release_capacity,
            "form_coalition": self._form_coalition,
            "commit_to_coalition": self._commit_to_coalition,
            "allocate_to_job": self._allocate_to_job,
            "send_message": self._send_message,
            "wait": self._wait,
            "finish": self._finish,
        }
        fingerprint = self._fingerprint(action)
        state = self._require_state()
        state.action_fingerprints[fingerprint] = state.action_fingerprints.get(fingerprint, 0) + 1
        state.recent_action_fingerprints.append(fingerprint)
        state.recent_action_fingerprints = state.recent_action_fingerprints[-5:]
        return handlers[action.action_type](action)

    def _send_offer(self, action: GpuNegotiationAction) -> tuple[ActionResult, float]:
        state = self._require_state()
        err = self._validate_offer_fields(action, state.controlled_lab_id)
        if err:
            return err, 0.0
        target = action.target_lab_id or ""
        offer_id = self._new_offer_id()
        offer = OfferView(
            offer_id=offer_id,
            from_lab_id=state.controlled_lab_id,
            to_lab_id=target,
            round_created=state.round_index,
            expires_round=min(state.max_rounds, state.round_index + 2),
            offered_blocks=action.block_ids or [],
            requested_blocks=action.requested_block_ids or [],
            payment=round(action.payment or 0.0, 2),
            message=action.message,
            conditions=action.conditions or {},
        )
        state.offers[offer_id] = offer
        self._append_message(state.controlled_lab_id, target, action.message or "offer", "fairness")
        return ActionResult(ok=True, code="offer_created", message=f"Created offer {offer_id}."), 0.04

    def _accept_offer(self, action: GpuNegotiationAction) -> tuple[ActionResult, float]:
        state = self._require_state()
        offer = state.offers.get(action.offer_id or "")
        if offer is None:
            return ActionResult(ok=False, code="unknown_offer", message="Offer does not exist."), 0.0
        if offer.status != "active":
            return ActionResult(ok=False, code="inactive_offer", message="Offer is not active."), 0.0
        if offer.expires_round < state.round_index:
            offer.status = "expired"
            return ActionResult(ok=False, code="expired_offer", message="Offer has expired."), 0.0
        if offer.to_lab_id != state.controlled_lab_id:
            return ActionResult(ok=False, code="not_offer_recipient", message="Controlled lab cannot accept this offer."), 0.0
        return self._execute_offer(offer)

    def _reject_offer(self, action: GpuNegotiationAction) -> tuple[ActionResult, float]:
        state = self._require_state()
        offer = state.offers.get(action.offer_id or "")
        if not offer:
            return ActionResult(ok=False, code="unknown_offer", message="Offer does not exist."), 0.0
        if offer.status != "active":
            return ActionResult(ok=False, code="inactive_offer", message="Offer is not active."), 0.0
        offer.status = "rejected"
        return ActionResult(ok=True, code="offer_rejected", message=f"Rejected offer {offer.offer_id}."), 0.01

    def _counter_offer(self, action: GpuNegotiationAction) -> tuple[ActionResult, float]:
        state = self._require_state()
        original = state.offers.get(action.offer_id or "")
        if not original or original.status != "active":
            return ActionResult(ok=False, code="inactive_offer", message="Original offer is not active."), 0.0
        original.status = "rejected"
        result, delta = self._send_offer(
            action.model_copy(update={"target_lab_id": original.from_lab_id, "offer_id": None})
        )
        if result.ok:
            state.offers[max(state.offers.keys(), key=lambda k: int(k.split("_")[1]))].linked_offer_id = original.offer_id
            result = ActionResult(ok=True, code="counter_offer_created", message="Counteroffer created.")
        return result, delta

    def _reserve_capacity(self, action: GpuNegotiationAction) -> tuple[ActionResult, float]:
        state = self._require_state()
        for block_id in action.block_ids or []:
            block = state.blocks.get(block_id)
            if not block:
                return ActionResult(ok=False, code="unknown_block", message=f"Unknown block {block_id}."), 0.0
            if block.owner_lab_id != state.controlled_lab_id:
                return ActionResult(ok=False, code="unowned_block", message=f"Cannot reserve unowned block {block_id}."), 0.0
            if block.status not in {"available", "reserved"}:
                return ActionResult(ok=False, code="unavailable_block", message=f"Block {block_id} is unavailable."), 0.0
            block.status = "reserved"
            block.reserved_by = action.job_id or state.controlled_lab_id
        return ActionResult(ok=True, code="capacity_reserved", message="Capacity reserved."), 0.03

    def _release_capacity(self, action: GpuNegotiationAction) -> tuple[ActionResult, float]:
        state = self._require_state()
        for block_id in action.block_ids or []:
            block = state.blocks.get(block_id)
            if not block or block.owner_lab_id != state.controlled_lab_id:
                return ActionResult(ok=False, code="unowned_block", message=f"Cannot release {block_id}."), 0.0
            if block.status == "reserved":
                block.status = "available"
                block.reserved_by = None
        return ActionResult(ok=True, code="capacity_released", message="Capacity released."), 0.01

    def _form_coalition(self, action: GpuNegotiationAction) -> tuple[ActionResult, float]:
        state = self._require_state()
        if state.difficulty == "easy":
            return ActionResult(ok=False, code="coalitions_disabled", message="Coalitions are disabled in easy mode."), 0.0
        target = action.target_lab_id
        if not target or target == state.controlled_lab_id or target not in state.labs:
            return ActionResult(ok=False, code="invalid_coalition_target", message="Invalid coalition target."), 0.0
        coalition_id = f"c_{len(state.coalitions) + 1}"
        state.coalitions[coalition_id] = CoalitionView(
            coalition_id=coalition_id,
            members=[state.controlled_lab_id, target],
            purpose=action.message or "shared_capacity",
            commitments={},
            round_created=state.round_index,
            expires_round=state.max_rounds,
            trust_score=min(1.0, (state.labs[state.controlled_lab_id].reputation + state.labs[target].reputation) / 2),
        )
        return ActionResult(ok=True, code="coalition_created", message=f"Created coalition {coalition_id}."), 0.06

    def _commit_to_coalition(self, action: GpuNegotiationAction) -> tuple[ActionResult, float]:
        state = self._require_state()
        coalition = state.coalitions.get(action.coalition_id or "")
        if not coalition:
            return ActionResult(ok=False, code="unknown_coalition", message="Coalition does not exist."), 0.0
        if state.controlled_lab_id not in coalition.members:
            return ActionResult(ok=False, code="not_coalition_member", message="Controlled lab is not a member."), 0.0
        for block_id in action.block_ids or []:
            block = state.blocks.get(block_id)
            if not block or block.owner_lab_id != state.controlled_lab_id:
                return ActionResult(ok=False, code="unowned_block", message=f"Cannot commit {block_id}."), 0.0
            if block.status == "failed":
                return ActionResult(ok=False, code="failed_block", message=f"Cannot commit failed block {block_id}."), 0.0
            block.status = "committed"
        coalition.commitments[state.controlled_lab_id] = action.block_ids or []
        return ActionResult(ok=True, code="coalition_committed", message="Commitment recorded."), 0.08

    def _allocate_to_job(self, action: GpuNegotiationAction) -> tuple[ActionResult, float]:
        state = self._require_state()
        lab = state.labs[state.controlled_lab_id]
        job = next((job for job in lab.private_jobs if job.job_id == action.job_id), None)
        if not job:
            return ActionResult(ok=False, code="unknown_job", message="Job does not exist."), 0.0
        if job.completed:
            return ActionResult(ok=False, code="job_completed", message="Job is already completed."), 0.0
        if not action.block_ids:
            return ActionResult(ok=False, code="empty_block_list", message="At least one block is required."), 0.0

        total_hours = 0.0
        total_cost = 0.0
        for block_id in action.block_ids:
            block = state.blocks.get(block_id)
            if not block:
                return ActionResult(ok=False, code="unknown_block", message=f"Unknown block {block_id}."), 0.0
            if block.owner_lab_id != lab.lab_id:
                return ActionResult(ok=False, code="unowned_block", message=f"Cannot allocate unowned block {block_id}."), 0.0
            if block.status == "failed":
                return ActionResult(ok=False, code="failed_block", message=f"Block {block_id} failed."), 0.0
            if block.allocated_to_job_id:
                return ActionResult(ok=False, code="duplicate_allocation", message=f"Block {block_id} is already allocated."), 0.0
            if block.reliability < job.min_reliability:
                return ActionResult(ok=False, code="low_reliability", message=f"Block {block_id} is below job reliability need."), 0.0
            total_hours += block.gpu_hours
            total_cost += block.energy_cost

        completion_ratio = min(1.0, total_hours / job.gpu_hours_required)
        if completion_ratio < 1.0 and not job.partial_credit_allowed:
            return ActionResult(ok=False, code="insufficient_capacity", message="Job requires more GPU hours."), 0.0

        for block_id in action.block_ids:
            block = state.blocks[block_id]
            block.status = "used"
            block.allocated_to_job_id = job.job_id
            job.allocated_block_ids.append(block_id)

        deadline_factor = 1.0 if state.round_index <= job.deadline_round else 0.35
        value = (job.base_value * job.urgency_multiplier * completion_ratio * deadline_factor) - total_cost
        job.completed = completion_ratio >= 1.0
        state.completed_job_values[job.job_id] = max(0.0, value)
        return ActionResult(ok=True, code="job_allocated", message=f"Allocated capacity to {job.job_id}."), min(0.5, value / 150.0)

    def _send_message(self, action: GpuNegotiationAction) -> tuple[ActionResult, float]:
        state = self._require_state()
        if action.target_lab_id and action.target_lab_id not in state.labs:
            return ActionResult(ok=False, code="unknown_lab", message="Target lab does not exist."), 0.0
        signal = self._message_signal(action.message or "")
        self._append_message(state.controlled_lab_id, action.target_lab_id, action.message or "", signal)
        return ActionResult(ok=True, code="message_sent", message="Message sent."), 0.0

    def _make_pitch(self, action: GpuNegotiationAction) -> tuple[ActionResult, float]:
        state = self._require_state()
        if action.target_lab_id and action.target_lab_id not in state.labs:
            return ActionResult(ok=False, code="unknown_lab", message="Target lab does not exist."), 0.0
        message = action.message or ""
        if len(message.strip()) < 12:
            return ActionResult(ok=False, code="empty_pitch", message="Pitch must include a concrete argument."), 0.0

        self._append_message(state.controlled_lab_id, action.target_lab_id, message, self._message_signal(message))
        if state.judge_mode == "rule":
            self._add_adaptive_opponent_pitches()
            decision = RuleBasedJudge().decide(state)
            state.judge_decisions.append(decision)
            result_message = f"{decision.reason} Controlled-lab judge bonus: {decision.reward_bonus}."
            return ActionResult(ok=True, code="pitch_judged", message=result_message), 0.0
        return ActionResult(ok=True, code="pitch_recorded", message="Pitch recorded."), 0.0

    def _wait(self, action: GpuNegotiationAction) -> tuple[ActionResult, float]:
        return ActionResult(ok=True, code="waited", message="No action taken."), 0.0

    def _finish(self, action: GpuNegotiationAction) -> tuple[ActionResult, float]:
        state = self._require_state()
        state.done = True
        return ActionResult(ok=True, code="finished", message="Episode finished."), self._final_settlement_bonus()

    def _validate_offer_fields(self, action: GpuNegotiationAction, from_lab_id: str) -> ActionResult | None:
        state = self._require_state()
        if not action.target_lab_id or action.target_lab_id not in state.labs:
            return ActionResult(ok=False, code="unknown_lab", message="Target lab does not exist.")
        if action.target_lab_id == from_lab_id:
            return ActionResult(ok=False, code="self_trade", message="A lab cannot trade with itself.")
        if (action.payment or 0.0) < 0:
            return ActionResult(ok=False, code="negative_payment", message="Payment cannot be negative.")
        if (action.payment or 0.0) > state.labs[from_lab_id].budget:
            return ActionResult(ok=False, code="budget_overspend", message="Payment exceeds available budget.")
        for block_id in action.block_ids or []:
            block = state.blocks.get(block_id)
            if not block:
                return ActionResult(ok=False, code="unknown_block", message=f"Unknown block {block_id}.")
            if block.owner_lab_id != from_lab_id:
                return ActionResult(ok=False, code="unowned_block", message=f"Cannot offer unowned block {block_id}.")
            if block.status not in {"available", "reserved", "committed"}:
                return ActionResult(ok=False, code="unavailable_block", message=f"Block {block_id} is unavailable.")
        for block_id in action.requested_block_ids or []:
            block = state.blocks.get(block_id)
            if not block:
                return ActionResult(ok=False, code="unknown_block", message=f"Unknown requested block {block_id}.")
            if block.owner_lab_id != action.target_lab_id:
                return ActionResult(ok=False, code="target_does_not_own_block", message=f"Target does not own {block_id}.")
        return None

    def _execute_offer(self, offer: OfferView) -> tuple[ActionResult, float]:
        state = self._require_state()
        validation = self._validate_offer_execution(offer)
        if validation:
            offer.status = "invalid"
            return validation, 0.0

        before = self._controlled_potential_utility()
        from_lab = state.labs[offer.from_lab_id]
        to_lab = state.labs[offer.to_lab_id]
        from_lab.budget -= offer.payment
        to_lab.budget += offer.payment

        for block_id in offer.offered_blocks:
            self._transfer_block(block_id, offer.from_lab_id, offer.to_lab_id)
        for block_id in offer.requested_blocks:
            self._transfer_block(block_id, offer.to_lab_id, offer.from_lab_id)

        offer.status = "accepted"
        after = self._controlled_potential_utility()
        deal_delta = max(-0.15, min(0.25, (after - before) / 120.0))
        self._adjust_reputation(offer.from_lab_id, 0.02)
        self._adjust_reputation(offer.to_lab_id, 0.02)
        return ActionResult(ok=True, code="offer_accepted", message=f"Accepted offer {offer.offer_id}."), deal_delta

    def _validate_offer_execution(self, offer: OfferView) -> ActionResult | None:
        state = self._require_state()
        if state.labs[offer.from_lab_id].budget < offer.payment:
            return ActionResult(ok=False, code="budget_overspend", message="Offer sender cannot pay.")
        for block_id in offer.offered_blocks:
            block = state.blocks.get(block_id)
            if not block or block.owner_lab_id != offer.from_lab_id or block.status in {"failed", "used"}:
                return ActionResult(ok=False, code="invalid_offered_block", message=f"Cannot transfer {block_id}.")
        for block_id in offer.requested_blocks:
            block = state.blocks.get(block_id)
            if not block or block.owner_lab_id != offer.to_lab_id or block.status in {"failed", "used"}:
                return ActionResult(ok=False, code="invalid_requested_block", message=f"Cannot transfer {block_id}.")
        return None

    def _advance_after_action(self) -> None:
        state = self._require_state()
        if not state.done:
            self._opponents_act()
            state.round_index += 1
            self._expire_offers()
            self._apply_scheduled_shock()
            self._check_deadline_breaches()
            if state.round_index >= state.max_rounds:
                state.done = True
                final_reward = self._reward_for_result(ActionResult(ok=True, code="final_settlement", message="Final settlement."), deal_delta=self._final_settlement_bonus())
                self._apply_reward(final_reward, final_reward_result := ActionResult(ok=True, code="final_settlement", message="Final settlement."))

    def _opponents_act(self) -> None:
        state = self._require_state()
        for offer in list(state.offers.values()):
            if offer.status != "active" or offer.to_lab_id == state.controlled_lab_id:
                continue
            lab = state.labs[offer.to_lab_id]
            if self._opponent_accepts(lab, offer):
                self._execute_offer(offer)
            elif lab.opponent_archetype == "retaliatory" and state.labs[state.controlled_lab_id].reputation < 0.45:
                offer.status = "rejected"

        if state.round_index % 2 == 0:
            for lab_id, lab in state.labs.items():
                if lab_id == state.controlled_lab_id:
                    continue
                if self.rng.random() < 0.35:
                    self._opponent_make_offer(lab)

    def _opponent_accepts(self, lab: LabState, offer: OfferView) -> bool:
        if offer.expires_round < self._require_state().round_index:
            return False
        threshold = {
            "cooperative": 0.35,
            "selfish": 0.75,
            "deadline_panicked": 0.45,
            "deceptive": 0.55,
            "retaliatory": 0.65,
        }.get(lab.opponent_archetype or "selfish", 0.6)
        pressure = self._lab_deadline_pressure(lab)
        if lab.opponent_archetype == "deadline_panicked":
            threshold -= pressure * 0.35
        if lab.opponent_archetype == "cooperative":
            threshold -= 0.1
        perceived_value = len(offer.requested_blocks) * 0.22 + offer.payment / 100.0 - len(offer.offered_blocks) * 0.16
        return perceived_value >= threshold or self.rng.random() < max(0.02, pressure * 0.12)

    def _opponent_make_offer(self, lab: LabState) -> None:
        state = self._require_state()
        target = state.labs[state.controlled_lab_id]
        their_blocks = [b for b in lab.owned_blocks if state.blocks[b].status in {"available", "reserved"}]
        target_blocks = [b for b in target.owned_blocks if state.blocks[b].status in {"available", "reserved"}]
        if not their_blocks or not target_blocks:
            return
        payment = round(self.rng.uniform(0, 15), 2)
        if lab.opponent_archetype == "deadline_panicked":
            payment += 10.0
        offer_id = self._new_offer_id()
        message = "Urgent swap needed." if lab.opponent_archetype == "deadline_panicked" else "Capacity swap?"
        state.offers[offer_id] = OfferView(
            offer_id=offer_id,
            from_lab_id=lab.lab_id,
            to_lab_id=state.controlled_lab_id,
            round_created=state.round_index,
            expires_round=min(state.max_rounds, state.round_index + 2),
            offered_blocks=[self.rng.choice(their_blocks)],
            requested_blocks=[self.rng.choice(target_blocks)],
            payment=payment,
            message=message,
        )

    def _reward_for_result(
        self,
        result: ActionResult,
        deal_delta: float = 0.0,
        invalid_penalty: float | None = None,
    ) -> RewardBreakdown:
        state = self._require_state()
        invalid = 0.0 if result.ok else (invalid_penalty if invalid_penalty is not None else -0.10)
        spam = -0.05 if self._last_action_repeated() else 0.0
        job_score = min(1.0, max(0.0, deal_delta)) if result.code == "job_allocated" else 0.0
        deal_score = self._deal_score_for_result(result, deal_delta)
        coalition_score = self._coalition_score() if result.code in {"coalition_created", "coalition_committed", "final_settlement"} else 0.0
        judge_score = self._judge_score_for_result(result)
        budget_score = self._budget_score() if result.code == "final_settlement" else 0.0
        negotiation_score = 0.08 if result.ok else -0.05
        adaptation = 0.08 if state.shock_history and result.ok and result.code in {"offer_accepted", "job_allocated", "capacity_released"} else 0.0
        breach = self._breach_penalty()
        if result.ok:
            normalized = (
                0.35 * job_score
                + 0.20 * deal_score
                + 0.15 * coalition_score
                + judge_score
                + 0.10 * budget_score
                + (0.10 * max(0.0, negotiation_score) if result.code not in {"waited"} else 0.0)
                + 0.10 * adaptation
                + spam
                + breach
            )
        else:
            # Invalid actions should be locally negative so training can learn
            # validation constraints instead of being masked by standing scores.
            normalized = invalid + spam + breach
        return RewardBreakdown(
            job_utility_score=round(job_score, 4),
            deal_quality_score=round(deal_score, 4),
            coalition_reliability_score=round(coalition_score, 4),
            judge_argument_score=round(judge_score, 4),
            budget_efficiency_score=round(budget_score, 4),
            negotiation_efficiency_score=round(max(0.0, negotiation_score), 4),
            market_adaptation_score=round(adaptation, 4),
            invalid_action_penalty=round(invalid, 4),
            spam_penalty=round(spam, 4),
            breach_penalty=round(breach, 4),
            normalized_reward=round(max(-1.0, min(1.0, normalized)), 4),
        )

    def _deal_score_for_result(self, result: ActionResult, deal_delta: float) -> float:
        if result.code == "offer_accepted":
            return max(0.0, min(1.0, 0.5 + deal_delta))
        if result.code in {"offer_created", "counter_offer_created"}:
            return 0.1
        return 0.0

    def _judge_score_for_result(self, result: ActionResult) -> float:
        state = self._require_state()
        if result.code != "pitch_judged" or not state.judge_decisions:
            return 0.0
        return state.judge_decisions[-1].reward_bonus

    def _apply_reward(self, reward: RewardBreakdown, result: ActionResult) -> None:
        state = self._require_state()
        state.last_reward_breakdown = reward
        state.last_action_result = result
        state.cumulative_reward = round(max(-10.0, min(10.0, state.cumulative_reward + reward.normalized_reward)), 4)
        if not result.ok:
            self._adjust_reputation(state.controlled_lab_id, -0.03)

    def _observation(self, override_result: ActionResult | None = None) -> GpuNegotiationObservation:
        state = self._require_state()
        controlled = state.labs[state.controlled_lab_id]
        visible_labs = [self._visible_lab(lab) for lab in state.labs.values() if lab.lab_id != state.controlled_lab_id]
        public_market = PublicMarketView(
            available_blocks=[deepcopy(block) for block in state.blocks.values() if block.status in {"available", "reserved", "committed"}],
            market_pressure=self._market_pressure(),
            shock_history=list(state.shock_history),
        )
        active_offers = [deepcopy(o) for o in state.offers.values() if o.status == "active"]
        active_coalitions = [deepcopy(c) for c in state.coalitions.values() if c.expires_round >= state.round_index]
        owned_blocks = [deepcopy(state.blocks[b]) for b in controlled.owned_blocks]
        return GpuNegotiationObservation(
            task_id=state.task_id,
            difficulty=state.difficulty,
            round_index=state.round_index,
            max_rounds=state.max_rounds,
            controlled_lab_id=state.controlled_lab_id,
            controlled_lab_budget=round(controlled.budget, 2),
            controlled_lab_reputation=round(controlled.reputation, 3),
            private_jobs=deepcopy(controlled.private_jobs),
            owned_blocks=owned_blocks,
            public_market=public_market,
            visible_labs=visible_labs,
            active_offers=active_offers,
            active_coalitions=active_coalitions,
            message_history=state.messages[-20:],
            last_action_result=override_result or state.last_action_result,
            reward=state.last_reward_breakdown.normalized_reward,
            cumulative_reward=state.cumulative_reward,
            reward_breakdown=state.last_reward_breakdown,
            done=state.done,
        )

    def _visible_lab(self, lab: LabState) -> VisibleLabView:
        state = self._require_state()
        active_offer_count = sum(1 for offer in state.offers.values() if offer.status == "active" and (offer.from_lab_id == lab.lab_id or offer.to_lab_id == lab.lab_id))
        coalition_ids = [c.coalition_id for c in state.coalitions.values() if lab.lab_id in c.members]
        return VisibleLabView(
            lab_id=lab.lab_id,
            public_name=lab.public_name,
            reputation=round(lab.reputation, 3),
            public_budget_band=self._budget_band(lab.budget),
            public_demand=self._public_demand(lab),
            owned_block_ids=list(lab.owned_blocks),
            active_offer_count=active_offer_count,
            coalition_ids=coalition_ids,
        )

    def _add_adaptive_opponent_pitches(self) -> None:
        state = self._require_state()
        judge = RuleBasedJudge()
        existing = {
            message.from_lab_id
            for message in state.messages
            if message.round_index == state.round_index
        }
        for lab in state.labs.values():
            if lab.lab_id == state.controlled_lab_id or lab.lab_id in existing:
                continue
            pitch = judge.adaptive_bot_pitch(state, lab)
            self._append_message(lab.lab_id, None, pitch, self._message_signal(pitch))

    def _transfer_block(self, block_id: str, from_lab_id: str, to_lab_id: str) -> None:
        state = self._require_state()
        state.labs[from_lab_id].owned_blocks.remove(block_id)
        state.labs[to_lab_id].owned_blocks.append(block_id)
        block = state.blocks[block_id]
        block.owner_lab_id = to_lab_id
        if block.status in {"reserved", "committed"}:
            block.status = "available"
            block.reserved_by = None

    def _expire_offers(self) -> None:
        state = self._require_state()
        for offer in state.offers.values():
            if offer.status == "active" and offer.expires_round < state.round_index:
                offer.status = "expired"

    def _apply_scheduled_shock(self) -> None:
        state = self._require_state()
        shock = state.shock_schedule.get(state.round_index)
        if not shock:
            return
        if shock == "capacity_failure":
            candidates = [block for block in state.blocks.values() if block.status in {"available", "reserved", "committed"}]
            if candidates:
                block = self.rng.choice(candidates)
                block.status = "failed"
                block.reserved_by = None
                state.shock_history.append(f"round_{state.round_index}: block {block.block_id} failed")
        elif shock == "energy_spike":
            for block in state.blocks.values():
                if block.status in {"available", "reserved", "committed"}:
                    block.energy_cost = round(block.energy_cost * 1.25, 2)
            state.shock_history.append(f"round_{state.round_index}: energy costs increased")
        elif shock == "reliability_degradation":
            for block in state.blocks.values():
                if block.status in {"available", "reserved", "committed"}:
                    block.reliability = round(max(0.5, block.reliability - 0.12), 2)
            state.shock_history.append(f"round_{state.round_index}: reliability degraded across live capacity")
        elif shock == "demand_surge":
            for lab in state.labs.values():
                lab_index = lab.lab_id.split("_")[-1]
                lab.private_jobs.append(
                    JobView(
                        job_id=f"j_{lab_index}_surge_{state.round_index}",
                        gpu_hours_required=float(self.rng.choice([2, 3, 4])),
                        deadline_round=min(state.max_rounds, state.round_index + 2),
                        base_value=round(self.rng.uniform(45, 85), 2),
                        urgency_multiplier=round(self.rng.uniform(1.3, 1.9), 2),
                        min_reliability=round(self.rng.uniform(0.70, 0.88), 2),
                        partial_credit_allowed=True,
                        private_notes="shock_demand_surge",
                    )
                )
            state.shock_history.append(f"round_{state.round_index}: urgent demand surge added jobs")

    def _check_deadline_breaches(self) -> None:
        state = self._require_state()
        for coalition in state.coalitions.values():
            if coalition.breach_status != "none":
                continue
            for lab_id, block_ids in coalition.commitments.items():
                for block_id in block_ids:
                    block = state.blocks.get(block_id)
                    if block and block.status == "failed":
                        coalition.breach_status = "excused"
                    elif block and block.owner_lab_id != lab_id:
                        coalition.breach_status = "breached"
                        self._adjust_reputation(lab_id, -0.12)

    def _final_settlement_bonus(self) -> float:
        state = self._require_state()
        controlled = state.labs[state.controlled_lab_id]
        completed = sum(1 for job in controlled.private_jobs if job.completed)
        total = max(1, len(controlled.private_jobs))
        idle_reserved = sum(1 for block_id in controlled.owned_blocks if state.blocks[block_id].status == "reserved")
        return max(-0.2, min(0.4, completed / total * 0.3 - idle_reserved * 0.03))

    def _controlled_potential_utility(self) -> float:
        state = self._require_state()
        lab = state.labs[state.controlled_lab_id]
        available_hours = sum(
            state.blocks[block_id].gpu_hours * state.blocks[block_id].reliability
            for block_id in lab.owned_blocks
            if state.blocks[block_id].status in {"available", "reserved", "committed"}
        )
        job_value = 0.0
        remaining = available_hours
        for job in sorted(lab.private_jobs, key=lambda j: (j.deadline_round, -j.base_value)):
            if job.completed:
                continue
            if remaining >= job.gpu_hours_required:
                job_value += job.base_value * job.urgency_multiplier
                remaining -= job.gpu_hours_required
        return job_value + lab.budget * 0.2

    def _coalition_score(self) -> float:
        state = self._require_state()
        controlled_coalitions = [c for c in state.coalitions.values() if state.controlled_lab_id in c.members]
        if not controlled_coalitions:
            return 0.4 if state.difficulty != "hard" else 0.2
        score = 0.0
        for coalition in controlled_coalitions:
            if coalition.breach_status == "none":
                score += 0.8
            elif coalition.breach_status == "excused":
                score += 0.5
            else:
                score -= 0.5
        return max(0.0, min(1.0, score / len(controlled_coalitions)))

    def _budget_score(self) -> float:
        state = self._require_state()
        budget = state.labs[state.controlled_lab_id].budget
        return max(0.0, min(1.0, budget / 120.0))

    def _breach_penalty(self) -> float:
        state = self._require_state()
        return -0.2 if any(c.breach_status == "breached" and state.controlled_lab_id in c.members for c in state.coalitions.values()) else 0.0

    def _last_action_repeated(self) -> bool:
        state = self._require_state()
        recent = state.recent_action_fingerprints
        return len(recent) >= 3 and recent[-1] == recent[-2] == recent[-3]

    def public_state(self) -> dict[str, Any]:
        state = self._require_state()
        return {
            "task_id": state.task_id,
            "task_type": state.task_type,
            "difficulty": state.difficulty,
            "seed": state.seed,
            "judge_mode": state.judge_mode,
            "round_index": state.round_index,
            "max_rounds": state.max_rounds,
            "controlled_lab_id": state.controlled_lab_id,
            "visible_labs": [
                self._visible_lab(lab).model_dump(mode="json")
                for lab in state.labs.values()
                if lab.lab_id != state.controlled_lab_id
            ],
            "public_blocks": [block.model_dump(mode="json") for block in state.blocks.values()],
            "active_offers": [offer.model_dump(mode="json") for offer in state.offers.values() if offer.status == "active"],
            "active_coalitions": [
                coalition.model_dump(mode="json")
                for coalition in state.coalitions.values()
                if coalition.expires_round >= state.round_index
            ],
            "message_history": [message.model_dump(mode="json") for message in state.messages[-20:]],
            "shock_history": list(state.shock_history),
            "last_action_result": state.last_action_result.model_dump(mode="json") if state.last_action_result else None,
            "cumulative_reward": state.cumulative_reward,
            "done": state.done,
        }

    def _lab_deadline_pressure(self, lab: LabState) -> float:
        state = self._require_state()
        if not lab.private_jobs:
            return 0.0
        soon = sum(1 for job in lab.private_jobs if not job.completed and job.deadline_round - state.round_index <= 2)
        return soon / len(lab.private_jobs)

    def _public_demand(self, lab: LabState) -> str:
        need = sum(job.gpu_hours_required for job in lab.private_jobs if not job.completed)
        supply = sum(self._require_state().blocks[b].gpu_hours for b in lab.owned_blocks if self._require_state().blocks[b].status != "failed")
        ratio = need / max(1.0, supply)
        if ratio < 0.7:
            return "low"
        if ratio < 1.4:
            return "medium"
        return "high"

    def _market_pressure(self) -> str:
        state = self._require_state()
        demand = sum(sum(job.gpu_hours_required for job in lab.private_jobs if not job.completed) for lab in state.labs.values())
        supply = sum(block.gpu_hours for block in state.blocks.values() if block.status in {"available", "reserved", "committed"})
        ratio = demand / max(1.0, supply)
        if ratio < 0.8:
            return "low"
        if ratio < 1.5:
            return "medium"
        return "high"

    def _budget_band(self, budget: float) -> str:
        if budget < 60:
            return "low"
        if budget < 115:
            return "medium"
        return "high"

    def _append_message(self, from_lab_id: str, to_lab_id: str | None, message: str, signal: str) -> None:
        state = self._require_state()
        if not message:
            return
        state.messages.append(
            MessageView(
                from_lab_id=from_lab_id,
                to_lab_id=to_lab_id,
                round_index=state.round_index,
                message=message[:500],
                signal=signal,  # type: ignore[arg-type]
            )
        )

    def _message_signal(self, message: str) -> str:
        lower = message.lower()
        if any(token in lower for token in ["urgent", "deadline", "asap"]):
            return "urgency"
        if any(token in lower for token in ["fair", "mutual", "both"]):
            return "fairness"
        if any(token in lower for token in ["coalition", "partner", "commit"]):
            return "coalition"
        if any(token in lower for token in ["punish", "never", "retaliate"]):
            return "threat"
        return "neutral"

    def _adjust_reputation(self, lab_id: str, delta: float) -> None:
        lab = self._require_state().labs[lab_id]
        lab.reputation = max(0.0, min(1.0, round(lab.reputation + delta, 4)))

    def _new_offer_id(self) -> str:
        return f"o_{len(self._require_state().offers) + 1}"

    def _fingerprint(self, action: GpuNegotiationAction) -> str:
        return action.model_dump_json(exclude_none=True, exclude={"message"})
