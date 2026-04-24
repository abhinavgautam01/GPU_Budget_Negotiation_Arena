from __future__ import annotations

from gpu_budget_arena.models import GpuNegotiationAction, GpuNegotiationObservation


def _block_hours(block_ids: list[str], obs: GpuNegotiationObservation) -> float:
    blocks = {block.block_id: block for block in obs.public_market.available_blocks}
    return sum(blocks[block_id].gpu_hours * blocks[block_id].reliability for block_id in block_ids if block_id in blocks)


def _owned_block_ids(obs: GpuNegotiationObservation) -> set[str]:
    return {block.block_id for block in obs.owned_blocks}


def _usable_blocks_for_job(obs: GpuNegotiationObservation, job_id: str) -> list[str]:
    job = next(job for job in obs.private_jobs if job.job_id == job_id)
    return [
        block.block_id
        for block in obs.owned_blocks
        if block.status in {"available", "reserved", "committed"} and block.reliability >= job.min_reliability
    ]


def _incoming_offers(obs: GpuNegotiationObservation) -> list:
    return [offer for offer in obs.active_offers if offer.to_lab_id == obs.controlled_lab_id]


def _available_owned_block_ids(obs: GpuNegotiationObservation) -> set[str]:
    return {
        block.block_id
        for block in obs.owned_blocks
        if block.status in {"available", "reserved", "committed"} and block.allocated_to_job_id is None
    }


def random_validish_policy(obs: GpuNegotiationObservation) -> GpuNegotiationAction:
    incoming = _incoming_offers(obs)
    if incoming:
        return GpuNegotiationAction(action_type="reject_offer", offer_id=incoming[0].offer_id)
    return GpuNegotiationAction(action_type="wait")


def greedy_hoarder_policy(obs: GpuNegotiationObservation) -> GpuNegotiationAction:
    for job in obs.private_jobs:
        if not job.completed:
            usable = _usable_blocks_for_job(obs, job.job_id)
            if usable:
                return GpuNegotiationAction(action_type="allocate_to_job", job_id=job.job_id, block_ids=usable)
    return GpuNegotiationAction(action_type="finish")


def always_accept_policy(obs: GpuNegotiationObservation) -> GpuNegotiationAction:
    incoming = _incoming_offers(obs)
    if incoming:
        return GpuNegotiationAction(action_type="accept_offer", offer_id=incoming[0].offer_id)
    return greedy_hoarder_policy(obs)


def rule_based_expert_policy(obs: GpuNegotiationObservation) -> GpuNegotiationAction:
    owned_ids = _owned_block_ids(obs)
    available_owned_ids = _available_owned_block_ids(obs)

    for offer in _incoming_offers(obs):
        if not set(offer.requested_blocks).issubset(owned_ids):
            return GpuNegotiationAction(action_type="reject_offer", offer_id=offer.offer_id)
        if not set(offer.requested_blocks).issubset(available_owned_ids):
            return GpuNegotiationAction(action_type="reject_offer", offer_id=offer.offer_id)
        if any(block_id not in {block.block_id for block in obs.public_market.available_blocks} for block_id in offer.offered_blocks):
            return GpuNegotiationAction(action_type="reject_offer", offer_id=offer.offer_id)
        incoming_hours = _block_hours(offer.offered_blocks, obs)
        outgoing_hours = _block_hours(offer.requested_blocks, obs)
        if incoming_hours + offer.payment / 10.0 >= outgoing_hours * 0.95:
            return GpuNegotiationAction(action_type="accept_offer", offer_id=offer.offer_id)

    for job in sorted(obs.private_jobs, key=lambda item: (item.deadline_round, -item.base_value)):
        if job.completed:
            continue
        usable = _usable_blocks_for_job(obs, job.job_id)
        if _block_hours(usable, obs) >= job.gpu_hours_required:
            return GpuNegotiationAction(action_type="allocate_to_job", job_id=job.job_id, block_ids=usable)

    if obs.active_coalitions:
        coalition = obs.active_coalitions[0]
        if obs.owned_blocks:
            commit_candidates = [
                block.block_id
                for block in obs.owned_blocks
                if block.status in {"available", "reserved"} and not block.allocated_to_job_id
            ]
            if commit_candidates:
                return GpuNegotiationAction(
                    action_type="commit_to_coalition",
                    coalition_id=coalition.coalition_id,
                    block_ids=commit_candidates[:1],
                )

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
        offered = [
            block.block_id
            for block in obs.owned_blocks
            if block.status in {"available", "reserved"} and block.allocated_to_job_id is None
        ]
        if requested and offered:
            return GpuNegotiationAction(
                action_type="send_offer",
                target_lab_id=target.lab_id,
                block_ids=offered[:1],
                requested_block_ids=requested,
                payment=4.0 if obs.difficulty == "hard" else 2.0,
                message="fair swap to improve both deadline schedules",
            )

    return greedy_hoarder_policy(obs)
