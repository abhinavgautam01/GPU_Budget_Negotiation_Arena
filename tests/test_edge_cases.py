from __future__ import annotations

from gpu_budget_arena.env import GpuBudgetNegotiationEnv
from gpu_budget_arena.models import GpuNegotiationAction, OfferView, ResetConfig


def test_negative_payment_offer_is_rejected() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="market_round", seed=50))
    block_id = obs.owned_blocks[0].block_id
    target = obs.visible_labs[0].lab_id
    result = env.step(
        GpuNegotiationAction(
            action_type="send_offer",
            target_lab_id=target,
            block_ids=[block_id],
            payment=-1.0,
        )
    )
    assert result.last_action_result is not None
    assert result.last_action_result.code == "negative_payment"
    assert result.reward < 0


def test_requesting_block_target_does_not_own_is_rejected() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="market_round", seed=51))
    target = obs.visible_labs[0].lab_id
    bad_requested = obs.owned_blocks[0].block_id
    offered = obs.owned_blocks[1].block_id
    result = env.step(
        GpuNegotiationAction(
            action_type="send_offer",
            target_lab_id=target,
            block_ids=[offered],
            requested_block_ids=[bad_requested],
            payment=1.0,
        )
    )
    assert result.last_action_result is not None
    assert result.last_action_result.code == "target_does_not_own_block"


def test_reserve_then_release_owned_capacity_round_trips_state() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="single_trade", seed=52))
    block_id = obs.owned_blocks[0].block_id
    obs = env.step(GpuNegotiationAction(action_type="reserve_capacity", block_ids=[block_id], job_id=obs.private_jobs[0].job_id))
    assert obs.last_action_result is not None
    assert obs.last_action_result.code == "capacity_reserved"
    assert any(block.block_id == block_id and block.status == "reserved" for block in obs.owned_blocks)
    obs = env.step(GpuNegotiationAction(action_type="release_capacity", block_ids=[block_id]))
    assert obs.last_action_result is not None
    assert obs.last_action_result.code == "capacity_released"
    assert any(block.block_id == block_id and block.status == "available" for block in obs.owned_blocks)


def test_reserving_unowned_block_fails() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="market_round", seed=53))
    foreign_block = obs.visible_labs[0].owned_block_ids[0]
    obs = env.step(GpuNegotiationAction(action_type="reserve_capacity", block_ids=[foreign_block]))
    assert obs.last_action_result is not None
    assert obs.last_action_result.code == "unowned_block"


def test_releasing_unowned_block_fails() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="market_round", seed=54))
    foreign_block = obs.visible_labs[0].owned_block_ids[0]
    obs = env.step(GpuNegotiationAction(action_type="release_capacity", block_ids=[foreign_block]))
    assert obs.last_action_result is not None
    assert obs.last_action_result.code == "unowned_block"


def test_send_message_to_unknown_lab_fails() -> None:
    env = GpuBudgetNegotiationEnv()
    env.reset(ResetConfig(task_type="single_trade", seed=55))
    obs = env.step(GpuNegotiationAction(action_type="send_message", target_lab_id="missing", message="hello"))
    assert obs.last_action_result is not None
    assert obs.last_action_result.code == "unknown_lab"


def test_send_message_records_urgency_signal() -> None:
    env = GpuBudgetNegotiationEnv()
    env.reset(ResetConfig(task_type="single_trade", seed=56))
    obs = env.step(GpuNegotiationAction(action_type="send_message", message="urgent deadline asap"))
    assert obs.last_action_result is not None
    assert obs.last_action_result.code == "message_sent"
    assert obs.message_history
    assert obs.message_history[-1].signal == "urgency"


def test_coalitions_disabled_in_easy_mode() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="single_trade", seed=57))
    obs = env.step(GpuNegotiationAction(action_type="form_coalition", target_lab_id=obs.visible_labs[0].lab_id))
    assert obs.last_action_result is not None
    assert obs.last_action_result.code == "coalitions_disabled"


def test_invalid_coalition_target_is_rejected() -> None:
    env = GpuBudgetNegotiationEnv()
    env.reset(ResetConfig(task_type="coalition_market", seed=58))
    obs = env.step(GpuNegotiationAction(action_type="form_coalition", target_lab_id="missing"))
    assert obs.last_action_result is not None
    assert obs.last_action_result.code == "invalid_coalition_target"


def test_unknown_coalition_commit_fails() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="coalition_market", seed=59))
    block_id = obs.owned_blocks[0].block_id
    obs = env.step(
        GpuNegotiationAction(
            action_type="commit_to_coalition",
            coalition_id="c_missing",
            block_ids=[block_id],
        )
    )
    assert obs.last_action_result is not None
    assert obs.last_action_result.code == "unknown_coalition"


def test_committing_unowned_block_to_coalition_fails() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="coalition_market", seed=60))
    target = obs.visible_labs[0].lab_id
    obs = env.step(GpuNegotiationAction(action_type="form_coalition", target_lab_id=target))
    coalition_id = obs.active_coalitions[0].coalition_id
    foreign_block = obs.visible_labs[0].owned_block_ids[0]
    obs = env.step(
        GpuNegotiationAction(
            action_type="commit_to_coalition",
            coalition_id=coalition_id,
            block_ids=[foreign_block],
        )
    )
    assert obs.last_action_result is not None
    assert obs.last_action_result.code == "unowned_block"


def test_unknown_job_fails_allocation() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="single_trade", seed=61))
    obs = env.step(GpuNegotiationAction(action_type="allocate_to_job", job_id="j_missing", block_ids=[obs.owned_blocks[0].block_id]))
    assert obs.last_action_result is not None
    assert obs.last_action_result.code == "unknown_job"


def test_empty_block_list_fails_allocation() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="single_trade", seed=62))
    job_id = obs.private_jobs[0].job_id
    obs = env.step(GpuNegotiationAction(action_type="allocate_to_job", job_id=job_id, block_ids=[]))
    assert obs.last_action_result is not None
    assert obs.last_action_result.code == "empty_block_list"


def test_low_reliability_block_fails_allocation() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="single_trade", seed=63))
    job_id = obs.private_jobs[0].job_id
    block_id = obs.owned_blocks[0].block_id
    env.state_data.blocks[block_id].reliability = 0.1  # type: ignore[union-attr]
    obs = env.step(GpuNegotiationAction(action_type="allocate_to_job", job_id=job_id, block_ids=[block_id]))
    assert obs.last_action_result is not None
    assert obs.last_action_result.code == "low_reliability"


def test_insufficient_capacity_fails_when_partial_credit_disabled() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="single_trade", seed=64))
    job_id = obs.private_jobs[0].job_id
    block_id = obs.owned_blocks[0].block_id
    env.state_data.labs[obs.controlled_lab_id].private_jobs[0].partial_credit_allowed = False  # type: ignore[union-attr]
    env.state_data.labs[obs.controlled_lab_id].private_jobs[0].gpu_hours_required = 999.0  # type: ignore[union-attr]
    env.state_data.blocks[block_id].reliability = 1.0  # type: ignore[union-attr]
    env.state_data.labs[obs.controlled_lab_id].private_jobs[0].min_reliability = 0.0  # type: ignore[union-attr]
    obs = env.step(GpuNegotiationAction(action_type="allocate_to_job", job_id=job_id, block_ids=[block_id]))
    assert obs.last_action_result is not None
    assert obs.last_action_result.code == "insufficient_capacity"


def test_finish_marks_episode_done_and_next_step_returns_episode_done() -> None:
    env = GpuBudgetNegotiationEnv()
    env.reset(ResetConfig(task_type="single_trade", seed=65))
    obs = env.step(GpuNegotiationAction(action_type="finish"))
    assert obs.done is True
    obs = env.step(GpuNegotiationAction(action_type="wait"))
    assert obs.last_action_result is not None
    assert obs.last_action_result.code == "episode_done"


def test_capacity_failure_shock_marks_failed_block_and_records_history() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="market_round", seed=66))
    for _ in range(3):
        obs = env.step(GpuNegotiationAction(action_type="wait"))
    assert obs.public_market.shock_history
    assert any("failed" in item for item in obs.public_market.shock_history)
    assert any(block.status == "failed" for block in env.state_data.blocks.values())  # type: ignore[union-attr]


def test_energy_spike_shock_increases_block_costs() -> None:
    env = GpuBudgetNegotiationEnv()
    env.reset(ResetConfig(task_type="coalition_market", seed=67))
    before = {block_id: block.energy_cost for block_id, block in env.state_data.blocks.items()}  # type: ignore[union-attr]
    obs = None
    for _ in range(6):
        obs = env.step(GpuNegotiationAction(action_type="wait"))
    assert obs is not None
    assert any("energy costs increased" in item for item in obs.public_market.shock_history)
    after = {block_id: block.energy_cost for block_id, block in env.state_data.blocks.items()}  # type: ignore[union-attr]
    assert any(after[block_id] > before[block_id] for block_id in before)


def test_counter_offer_rejects_original_and_links_new_offer() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="market_round", seed=68))
    original_id = obs.active_offers[0].offer_id
    target_lab = obs.active_offers[0].from_lab_id
    block_id = obs.owned_blocks[0].block_id
    requested = obs.visible_labs[0].owned_block_ids[0]
    obs = env.step(
        GpuNegotiationAction(
            action_type="counter_offer",
            offer_id=original_id,
            target_lab_id=target_lab,
            block_ids=[block_id],
            requested_block_ids=[requested],
            payment=1.0,
        )
    )
    assert obs.last_action_result is not None
    assert obs.last_action_result.code == "counter_offer_created"
    state = env.state()
    assert state["offers"][original_id]["status"] == "rejected"
    linked = [offer for offer in state["offers"].values() if offer["linked_offer_id"] == original_id]
    assert linked


def test_accept_offer_not_addressed_to_controlled_lab_fails() -> None:
    env = GpuBudgetNegotiationEnv()
    env.reset(ResetConfig(task_type="market_round", seed=69))
    state = env.state_data  # type: ignore[assignment]
    offer = OfferView(
        offer_id="manual_offer",
        from_lab_id=state.controlled_lab_id,
        to_lab_id="lab_1",
        round_created=state.round_index,
        expires_round=state.round_index + 1,
        offered_blocks=[],
        requested_blocks=[],
        payment=0.0,
    )
    state.offers[offer.offer_id] = offer
    obs = env.step(GpuNegotiationAction(action_type="accept_offer", offer_id=offer.offer_id))
    assert obs.last_action_result is not None
    assert obs.last_action_result.code == "not_offer_recipient"


def test_message_length_validation_returns_malformed_action() -> None:
    env = GpuBudgetNegotiationEnv()
    env.reset(ResetConfig(task_type="single_trade", seed=70))
    obs = env.step({"action_type": "send_message", "message": "x" * 501})
    assert obs.last_action_result is not None
    assert obs.last_action_result.code == "malformed_action"


def test_repeated_waits_trigger_spam_penalty() -> None:
    env = GpuBudgetNegotiationEnv()
    env.reset(ResetConfig(task_type="single_trade", seed=71))
    obs = None
    for _ in range(3):
        obs = env.step(GpuNegotiationAction(action_type="wait"))
    assert obs is not None
    assert obs.reward_breakdown.spam_penalty < 0


def test_breached_coalition_sets_penalty_and_status() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="coalition_market", seed=72))
    target = obs.visible_labs[0].lab_id
    obs = env.step(GpuNegotiationAction(action_type="form_coalition", target_lab_id=target))
    coalition_id = obs.active_coalitions[0].coalition_id
    block_id = obs.owned_blocks[0].block_id
    obs = env.step(GpuNegotiationAction(action_type="commit_to_coalition", coalition_id=coalition_id, block_ids=[block_id]))
    env._transfer_block(block_id, obs.controlled_lab_id, target)
    obs = env.step(GpuNegotiationAction(action_type="wait"))
    coalition = env.state()["coalitions"][coalition_id]
    assert coalition["breach_status"] == "breached"
    assert obs.reward_breakdown.breach_penalty <= 0


def test_failed_committed_block_creates_excused_breach() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="coalition_market", seed=73))
    target = obs.visible_labs[0].lab_id
    obs = env.step(GpuNegotiationAction(action_type="form_coalition", target_lab_id=target))
    coalition_id = obs.active_coalitions[0].coalition_id
    block_id = obs.owned_blocks[0].block_id
    obs = env.step(GpuNegotiationAction(action_type="commit_to_coalition", coalition_id=coalition_id, block_ids=[block_id]))
    env.state_data.blocks[block_id].status = "failed"  # type: ignore[union-attr]
    obs = env.step(GpuNegotiationAction(action_type="wait"))
    coalition = env.state()["coalitions"][coalition_id]
    assert coalition["breach_status"] == "excused"
