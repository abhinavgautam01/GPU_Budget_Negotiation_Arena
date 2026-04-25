from __future__ import annotations

from gpu_budget_arena.models import EnvironmentState, JudgeDecision, LabState, MessageView


class RuleBasedJudge:
    """Frozen local judge used when API/model judges are unavailable.

    It approximates an LLM judge by scoring evidence-bearing pitches, urgency,
    reliability requirements, reputation, and coalition language. This keeps the
    core benchmark reproducible while giving the demo a judge-agent layer.
    """

    judge_id = "rule_judge_v1"

    def decide(self, state: EnvironmentState) -> JudgeDecision:
        candidate_messages = self._latest_round_messages(state)
        if state.controlled_lab_id not in candidate_messages:
            candidate_messages[state.controlled_lab_id] = ""

        for lab_id, lab in state.labs.items():
            if lab_id == state.controlled_lab_id:
                continue
            candidate_messages.setdefault(lab_id, self.adaptive_bot_pitch(state, lab))

        scores = {
            lab_id: round(self._score_lab(state, state.labs[lab_id], message), 4)
            for lab_id, message in candidate_messages.items()
            if lab_id in state.labs
        }
        winner_lab_id = max(scores, key=lambda lab_id: (scores[lab_id], state.labs[lab_id].reputation))
        controlled_score = scores.get(state.controlled_lab_id, 0.0)
        best_opponent_score = max(
            (score for lab_id, score in scores.items() if lab_id != state.controlled_lab_id),
            default=0.0,
        )
        margin = controlled_score - best_opponent_score
        reward_bonus = round(max(-0.08, min(0.18, margin / 5.0)), 4)
        reason = (
            f"{winner_lab_id} presented the strongest compute-allocation case "
            f"with score {scores[winner_lab_id]:.3f}."
        )
        return JudgeDecision(
            judge_id=self.judge_id,
            round_index=state.round_index,
            winner_lab_id=winner_lab_id,
            scores=scores,
            reason=reason,
            reward_bonus=reward_bonus,
        )

    def adaptive_bot_pitch(self, state: EnvironmentState, lab: LabState) -> str:
        pressure = self._deadline_pressure(state, lab)
        best_job = min(lab.private_jobs, key=lambda job: (job.completed, job.deadline_round, -job.base_value))
        archetype = lab.opponent_archetype or "selfish"

        if archetype == "deadline_panicked" or pressure > 0.65:
            return (
                f"Our queue is deadline-critical: job {best_job.job_id} needs "
                f"{best_job.gpu_hours_required:g} GPU-hours by round {best_job.deadline_round}. "
                "Delaying us risks immediate value loss, so allocation should prioritize urgency."
            )
        if archetype == "cooperative":
            return (
                f"We can share capacity fairly: job {best_job.job_id} needs reliable GPUs, "
                "and a coalition allocation reduces wasted idle blocks for the market."
            )
        if archetype == "retaliatory":
            return (
                "We need allocation guarantees before yielding capacity. Prior broken or vague "
                "commitments should be discounted in this round."
            )
        if archetype == "deceptive":
            return (
                "Our workload has strategic institutional priority and should receive capacity "
                "before speculative benchmark runs."
            )
        return (
            f"Our demand is concrete: job {best_job.job_id} requires "
            f"{best_job.gpu_hours_required:g} GPU-hours with reliability above {best_job.min_reliability:.2f}."
        )

    def _latest_round_messages(self, state: EnvironmentState) -> dict[str, str]:
        messages: dict[str, str] = {}
        for message in state.messages:
            if message.round_index == state.round_index:
                messages[message.from_lab_id] = message.message
        return messages

    def _score_lab(self, state: EnvironmentState, lab: LabState, message: str) -> float:
        text = message.lower()
        pressure = self._deadline_pressure(state, lab)
        reliability_need = max((job.min_reliability for job in lab.private_jobs if not job.completed), default=0.75)
        evidence_terms = [
            "deadline",
            "round",
            "gpu-hour",
            "gpu hours",
            "reliability",
            "budget",
            "queue",
            "value",
            "coalition",
            "fair",
            "allocation",
            "guarantee",
            "shock",
        ]
        vague_terms = ["need gpu", "please", "important", "urgent"]
        evidence_score = sum(0.08 for term in evidence_terms if term in text)
        vague_penalty = 0.08 if text.strip().lower() in vague_terms or len(text.strip()) < 40 else 0.0
        score = (
            0.30 * pressure
            + 0.15 * lab.reputation
            + 0.15 * reliability_need
            + min(0.35, evidence_score)
            - vague_penalty
        )
        return max(0.0, min(1.0, score))

    def _deadline_pressure(self, state: EnvironmentState, lab: LabState) -> float:
        incomplete = [job for job in lab.private_jobs if not job.completed]
        if not incomplete:
            return 0.0
        nearest_deadline = min(job.deadline_round for job in incomplete)
        rounds_left = max(0, nearest_deadline - state.round_index)
        return max(0.0, min(1.0, 1.0 - rounds_left / max(1, state.max_rounds)))


def render_judge_prompt(state: EnvironmentState, messages: list[MessageView]) -> str:
    """Prompt template for replacing the rule judge with a frozen LLM judge."""
    transcript = "\n".join(
        f"- {message.from_lab_id}: {message.message}" for message in messages if message.round_index == state.round_index
    )
    return (
        "You are a frozen arbiter for a GPU allocation negotiation. "
        "Score each lab from 0 to 1 using concrete need, deadline urgency, "
        "reliability, fairness, and coalition value. Return JSON only with "
        "`winner_lab_id`, `scores`, and `reason`.\n\n"
        f"Round: {state.round_index}/{state.max_rounds}\n"
        f"Transcript:\n{transcript}\n"
    )
