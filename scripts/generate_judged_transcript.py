from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpu_budget_arena.env import GpuBudgetNegotiationEnv
from gpu_budget_arena.judge import render_judge_prompt
from gpu_budget_arena.models import GpuNegotiationAction, ResetConfig


def build_negotiator_pitch(obs: object) -> str:
    private_jobs = getattr(obs, "private_jobs")
    owned_blocks = getattr(obs, "owned_blocks")
    visible_labs = getattr(obs, "visible_labs")
    job = min(private_jobs, key=lambda item: (item.completed, item.deadline_round, -item.base_value))
    usable_blocks = [
        block
        for block in owned_blocks
        if block.status in {"available", "reserved", "committed"} and block.reliability >= job.min_reliability
    ]
    best_partner = max(visible_labs, key=lambda lab: lab.reputation) if visible_labs else None
    partner_clause = (
        f" I am willing to form a coalition with {best_partner.lab_id} for reciprocal capacity."
        if best_partner
        else ""
    )
    return (
        f"Lab 0 requests priority allocation for job {job.job_id}: it needs "
        f"{job.gpu_hours_required:g} GPU-hours by round {job.deadline_round}, "
        f"requires reliability >= {job.min_reliability:.2f}, and has value "
        f"{job.base_value:.1f} with urgency multiplier {job.urgency_multiplier:.2f}. "
        f"We currently have {len(usable_blocks)} compatible blocks and can avoid waste by "
        f"allocating immediately or trading lower-fit blocks fairly.{partner_clause}"
    )


def run_judged_episode(task_type: str, seed: int, max_pitches: int) -> tuple[GpuBudgetNegotiationEnv, list[dict[str, object]]]:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type=task_type, seed=seed, judge_mode="rule"))
    events: list[dict[str, object]] = []

    for _ in range(max_pitches):
        if obs.done:
            break
        pitch = build_negotiator_pitch(obs)
        obs = env.step(GpuNegotiationAction(action_type="make_pitch", message=pitch))
        state = env.state_data
        assert state is not None
        decision = state.judge_decisions[-1]
        round_messages = [
            message.model_dump(mode="json")
            for message in state.messages
            if message.round_index == decision.round_index
        ]
        events.append(
            {
                "round_index": decision.round_index,
                "controlled_pitch": pitch,
                "messages": round_messages,
                "judge_prompt": render_judge_prompt(state, state.messages),
                "judge_decision": decision.model_dump(mode="json"),
                "env_reward": obs.reward,
                "reward_breakdown": obs.reward_breakdown.model_dump(mode="json"),
            }
        )
    return env, events


def render_markdown(task_type: str, seed: int, events: list[dict[str, object]]) -> str:
    sections = [
        "# Judged Negotiation Transcript",
        "",
        f"- Task type: `{task_type}`",
        f"- Seed: `{seed}`",
        "- Judge mode: `rule`",
        "",
        "This transcript demonstrates the hybrid architecture: deterministic environment reward remains primary, while a frozen judge-agent scores natural-language negotiation quality.",
    ]
    for event in events:
        decision = event["judge_decision"]
        assert isinstance(decision, dict)
        sections.extend(
            [
                "",
                f"## Round {event['round_index']}",
                "",
                "### Pitches",
                "",
            ]
        )
        messages = event["messages"]
        assert isinstance(messages, list)
        for message in messages:
            assert isinstance(message, dict)
            sections.append(f"- `{message['from_lab_id']}`: {message['message']}")
        sections.extend(
            [
                "",
                "### Judge Decision",
                "",
                f"- Winner: `{decision['winner_lab_id']}`",
                f"- Scores: `{json.dumps(decision['scores'], sort_keys=True)}`",
                f"- Reason: {decision['reason']}",
                f"- Controlled-lab judge bonus: `{decision['reward_bonus']}`",
                f"- Environment reward after action: `{event['env_reward']}`",
                f"- Reward breakdown: `{json.dumps(event['reward_breakdown'], sort_keys=True)}`",
            ]
        )
    return "\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", default="coalition_market")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--max-pitches", type=int, default=3)
    parser.add_argument("--output", default="artifacts/judged_transcript.md")
    args = parser.parse_args()

    env, events = run_judged_episode(args.task_type, args.seed, args.max_pitches)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(args.task_type, args.seed, events), encoding="utf-8")
    print(
        {
            "output": str(output),
            "events": len(events),
            "judge_decisions": len(env.state_data.judge_decisions if env.state_data else []),
        }
    )


if __name__ == "__main__":
    main()
