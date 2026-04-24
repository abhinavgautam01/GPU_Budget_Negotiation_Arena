from __future__ import annotations

from gpu_budget_arena.env import GpuBudgetNegotiationEnv
from gpu_budget_arena.models import GpuNegotiationAction, ResetConfig


def test_reset_is_deterministic_for_same_seed() -> None:
    env_a = GpuBudgetNegotiationEnv()
    env_b = GpuBudgetNegotiationEnv()
    obs_a = env_a.reset(ResetConfig(task_type="market_round", seed=123))
    obs_b = env_b.reset(ResetConfig(task_type="market_round", seed=123))
    assert obs_a.model_dump(mode="json") == obs_b.model_dump(mode="json")


def test_reset_differs_for_different_seed() -> None:
    env_a = GpuBudgetNegotiationEnv()
    env_b = GpuBudgetNegotiationEnv()
    obs_a = env_a.reset(ResetConfig(task_type="market_round", seed=123))
    obs_b = env_b.reset(ResetConfig(task_type="market_round", seed=124))
    assert obs_a.private_jobs != obs_b.private_jobs


def test_invalid_self_trade_is_penalized_not_crashed() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="single_trade", seed=1))
    block = obs.owned_blocks[0]
    next_obs = env.step(
        GpuNegotiationAction(
            action_type="send_offer",
            target_lab_id=obs.controlled_lab_id,
            block_ids=[block.block_id],
            payment=1.0,
        )
    )
    assert next_obs.last_action_result is not None
    assert next_obs.last_action_result.ok is False
    assert next_obs.last_action_result.code == "self_trade"
    assert next_obs.reward < 0


def test_accepting_active_offer_transfers_assets_atomically() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="market_round", seed=2))
    offer = obs.active_offers[0]
    state_before = env.state()["labs"]
    controlled_blocks_before = set(state_before[obs.controlled_lab_id]["owned_blocks"])
    next_obs = env.step(GpuNegotiationAction(action_type="accept_offer", offer_id=offer.offer_id))
    assert next_obs.last_action_result is not None
    assert next_obs.last_action_result.ok is True
    state_after = env.state()
    controlled_blocks_after = set(state_after["labs"][obs.controlled_lab_id]["owned_blocks"])
    assert set(offer.offered_blocks).issubset(controlled_blocks_after)
    assert not set(offer.offered_blocks).issubset(controlled_blocks_before)
    assert state_after["offers"][offer.offer_id]["status"] == "accepted"


def test_allocate_to_job_marks_blocks_used_and_records_reward_columns() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="single_trade", seed=4))
    job = obs.private_jobs[0]
    usable = [block.block_id for block in obs.owned_blocks if block.reliability >= job.min_reliability]
    result = env.step(GpuNegotiationAction(action_type="allocate_to_job", job_id=job.job_id, block_ids=usable))
    assert result.reward_breakdown.normalized_reward <= 1.0
    assert result.reward_breakdown.normalized_reward >= -1.0
    assert result.last_action_result is not None
    assert result.last_action_result.code in {"job_allocated", "insufficient_capacity"}


def test_episode_terminates_within_max_rounds() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="coalition_market", seed=5))
    for _ in range(obs.max_rounds + 3):
        if obs.done:
            break
        obs = env.step(GpuNegotiationAction(action_type="wait"))
    assert obs.done is True
    assert obs.round_index <= obs.max_rounds

def test_observation_does_not_leak_opponent_private_jobs() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="coalition_market", seed=6))
    dumped = obs.model_dump(mode="json")
    assert "private_jobs" in dumped
    for visible_lab in dumped["visible_labs"]:
        assert "private_jobs" not in visible_lab

