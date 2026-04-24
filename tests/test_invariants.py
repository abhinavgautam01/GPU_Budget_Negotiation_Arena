from __future__ import annotations

from gpu_budget_arena.env import GpuBudgetNegotiationEnv
from gpu_budget_arena.models import GpuNegotiationAction, ResetConfig


def assert_invariants(env: GpuBudgetNegotiationEnv) -> None:
    state = env.state()
    owners = []
    for lab in state["labs"].values():
        assert lab["budget"] >= 0
        owners.extend(lab["owned_blocks"])
    assert len(owners) == len(set(owners))
    assert set(owners) == set(state["blocks"].keys())

    allocations = [
        block["allocated_to_job_id"]
        for block in state["blocks"].values()
        if block["allocated_to_job_id"] is not None
    ]
    assert len(allocations) == len(set(zip(allocations, [b for b in state["blocks"].keys()])))
    assert -10 <= state["cumulative_reward"] <= 10
    assert state["last_reward_breakdown"]["normalized_reward"] >= -1
    assert state["last_reward_breakdown"]["normalized_reward"] <= 1


def test_random_wait_episodes_preserve_conservation() -> None:
    for seed in range(10, 15):
        env = GpuBudgetNegotiationEnv()
        obs = env.reset(ResetConfig(task_type="coalition_market", seed=seed))
        assert_invariants(env)
        while not obs.done:
            obs = env.step(GpuNegotiationAction(action_type="wait"))
            assert_invariants(env)


def test_expired_offer_cannot_be_accepted() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="market_round", seed=20))
    offer_id = obs.active_offers[0].offer_id
    for _ in range(3):
        obs = env.step(GpuNegotiationAction(action_type="wait"))
    obs = env.step(GpuNegotiationAction(action_type="accept_offer", offer_id=offer_id))
    assert obs.last_action_result is not None
    assert obs.last_action_result.ok is False
    assert obs.last_action_result.code in {"inactive_offer", "expired_offer", "episode_done"}

