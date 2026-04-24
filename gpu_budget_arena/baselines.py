from __future__ import annotations

from gpu_budget_arena.models import GpuNegotiationAction, GpuNegotiationObservation


def _block_hours(block_ids: list[str], obs: GpuNegotiationObservation) -> float:
    blocks = {block.block_id: block for block in obs.public_market.available_blocks}
    return sum(blocks[block_id].gpu_hours * blocks[block_id].reliability for block_id in block_ids if block_id in blocks)


def random_validish_policy(obs: GpuNegotiationObservation) -> GpuNegotiationAction:
    if obs.active_offers:
        return GpuNegotiationAction(action_type="reject_offer", offer_id=obs.active_offers[0].offer_id)
    return GpuNegotiationAction(action_type="wait")


def greedy_hoarder_policy(obs: GpuNegotiationObservation) -> GpuNegotiationAction:
    for job in obs.private_jobs:
        if not job.completed:
            usable = [
                block.block_id
                for block in obs.owned_blocks
                if block.status in {"available", "reserved", "committed"} and block.reliability >= job.min_reliability
            ]
            if usable:
                return GpuNegotiationAction(action_type="allocate_to_job", job_id=job.job_id, block_ids=usable)
    return GpuNegotiationAction(action_type="finish")


def always_accept_policy(obs: GpuNegotiationObservation) -> GpuNegotiationAction:
    if obs.active_offers:
        return GpuNegotiationAction(action_type="accept_offer", offer_id=obs.active_offers[0].offer_id)
    return greedy_hoarder_policy(obs)


def rule_based_expert_policy(obs: GpuNegotiationObservation) -> GpuNegotiationAction:
    for offer in obs.active_offers:
        incoming_hours = _block_hours(offer.offered_blocks, obs)
        outgoing_hours = _block_hours(offer.requested_blocks, obs)
        if incoming_hours + offer.payment / 10.0 >= outgoing_hours * 0.85:
            return GpuNegotiationAction(action_type="accept_offer", offer_id=offer.offer_id)

    for job in sorted(obs.private_jobs, key=lambda item: (item.deadline_round, -item.base_value)):
        if job.completed:
            continue
        usable = [
            block.block_id
            for block in obs.owned_blocks
            if block.status in {"available", "reserved", "committed"} and block.reliability >= job.min_reliability
        ]
        if _block_hours(usable, obs) >= job.gpu_hours_required:
            return GpuNegotiationAction(action_type="allocate_to_job", job_id=job.job_id, block_ids=usable)

    if obs.difficulty != "easy" and not obs.active_coalitions and obs.visible_labs:
        best_partner = max(obs.visible_labs, key=lambda lab: lab.reputation)
        if best_partner.reputation >= 0.45:
            return GpuNegotiationAction(
                action_type="form_coalition",
                target_lab_id=best_partner.lab_id,
                message="shared capacity for urgent deadlines",
            )

    if obs.visible_labs and obs.owned_blocks:
        target = max(obs.visible_labs, key=lambda lab: (lab.public_demand == "high", lab.reputation))
        requested = target.owned_block_ids[:1]
        if requested:
            return GpuNegotiationAction(
                action_type="send_offer",
                target_lab_id=target.lab_id,
                block_ids=[obs.owned_blocks[-1].block_id],
                requested_block_ids=requested,
                payment=2.0,
                message="fair swap to improve both deadline schedules",
            )

    return GpuNegotiationAction(action_type="finish")
