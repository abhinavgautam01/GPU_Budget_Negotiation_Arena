from __future__ import annotations

from gpu_budget_arena.env import GpuBudgetNegotiationEnv
from gpu_budget_arena.models import GpuNegotiationAction, ResetConfig


def test_malformed_action_is_reported_through_observation() -> None:
    env = GpuBudgetNegotiationEnv()
    env.reset(ResetConfig(task_type="single_trade", seed=30))
    obs = env.step({"action_type": "not_real"})
    assert obs.last_action_result is not None
    assert obs.last_action_result.ok is False
    assert obs.last_action_result.code == "malformed_action"
    assert obs.reward < 0


def test_budget_overspend_offer_is_rejected() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="market_round", seed=31))
    block_id = obs.owned_blocks[0].block_id
    target = obs.visible_labs[0].lab_id
    result = env.step(
        GpuNegotiationAction(
            action_type="send_offer",
            target_lab_id=target,
            block_ids=[block_id],
            payment=10_000,
        )
    )
    assert result.last_action_result is not None
    assert result.last_action_result.code == "budget_overspend"
    assert result.reward_breakdown.invalid_action_penalty < 0


def test_coalition_commitment_tracks_blocks() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="coalition_market", seed=32))
    target = obs.visible_labs[0].lab_id
    obs = env.step(GpuNegotiationAction(action_type="form_coalition", target_lab_id=target, message="shared capacity"))
    assert obs.last_action_result is not None
    assert obs.last_action_result.ok is True
    coalition_id = obs.active_coalitions[0].coalition_id
    block_id = obs.owned_blocks[0].block_id
    obs = env.step(
        GpuNegotiationAction(
            action_type="commit_to_coalition",
            coalition_id=coalition_id,
            block_ids=[block_id],
        )
    )
    assert obs.last_action_result is not None
    assert obs.last_action_result.ok is True
    state = env.state()
    assert state["coalitions"][coalition_id]["commitments"][obs.controlled_lab_id] == [block_id]
    assert state["blocks"][block_id]["status"] == "committed"


def test_failed_block_cannot_be_allocated() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="market_round", seed=33))
    state = env.state()
    block_id = obs.owned_blocks[0].block_id
    state["blocks"][block_id]["status"] = "failed"
    # Mutate through the real state object for this focused validation case.
    env.state_data.blocks[block_id].status = "failed"  # type: ignore[union-attr]
    job_id = obs.private_jobs[0].job_id
    obs = env.step(GpuNegotiationAction(action_type="allocate_to_job", job_id=job_id, block_ids=[block_id]))
    assert obs.last_action_result is not None
    assert obs.last_action_result.code == "failed_block"

