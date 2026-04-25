from __future__ import annotations

from gpu_budget_arena.env import GpuBudgetNegotiationEnv
from gpu_budget_arena.judge import RuleBasedJudge
from gpu_budget_arena.models import GpuNegotiationAction, ResetConfig


def test_rule_judge_scores_pitches_without_replacing_core_reward() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="market_round", seed=80, judge_mode="rule"))
    pitch = (
        "We need 4 GPU-hours by round 2 with reliability above 0.85. "
        "A fair allocation avoids idle capacity and protects deadline value."
    )
    obs = env.step(GpuNegotiationAction(action_type="make_pitch", message=pitch))

    assert obs.last_action_result is not None
    assert obs.last_action_result.code == "pitch_judged"
    assert obs.reward_breakdown.judge_argument_score != 0.0
    assert env.state_data is not None
    assert env.state_data.judge_decisions
    assert env.state_data.judge_decisions[-1].scores


def test_pitch_without_judge_mode_is_recorded_only() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="market_round", seed=81))
    obs = env.step(
        GpuNegotiationAction(
            action_type="make_pitch",
            message="We request a fair GPU allocation for a deadline-bound benchmark job.",
        )
    )

    assert obs.last_action_result is not None
    assert obs.last_action_result.code == "pitch_recorded"
    assert obs.reward_breakdown.judge_argument_score == 0.0
    assert env.state_data is not None
    assert env.state_data.judge_decisions == []


def test_adaptive_bot_pitch_mentions_private_need_without_leaking_in_observation() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="coalition_market", seed=82, judge_mode="rule"))
    state = env.state_data
    assert state is not None
    opponent = state.labs[obs.visible_labs[0].lab_id]
    pitch = RuleBasedJudge().adaptive_bot_pitch(state, opponent)

    assert opponent.private_jobs[0].job_id in pitch or "allocation" in pitch
    dumped = obs.model_dump(mode="json")
    assert "private_jobs" not in dumped["visible_labs"][0]
