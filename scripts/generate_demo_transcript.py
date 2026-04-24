from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpu_budget_arena.baselines import (
    always_accept_policy,
    greedy_hoarder_policy,
    random_validish_policy,
    rule_based_expert_policy,
)
from gpu_budget_arena.env import GpuBudgetNegotiationEnv
from gpu_budget_arena.models import ResetConfig


POLICIES = {
    "random_validish": random_validish_policy,
    "greedy_hoarder": greedy_hoarder_policy,
    "always_accept": always_accept_policy,
    "rule_based_expert": rule_based_expert_policy,
}


def run_episode(task_type: str, seed: int, policy_name: str) -> tuple[object, list[dict[str, object]]]:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type=task_type, seed=seed))
    events: list[dict[str, object]] = []
    while not obs.done:
        action = POLICIES[policy_name](obs)
        obs = env.step(action)
        result = obs.last_action_result.model_dump() if obs.last_action_result else {}
        events.append(
            {
                "step": obs.round_index,
                "action": action.model_dump(exclude_none=True),
                "result": result,
                "reward": obs.reward,
                "cumulative_reward": obs.cumulative_reward,
            }
        )
    return obs, events


def render_markdown(task_type: str, seed: int, policy: str, final_reward: float, events: list[dict[str, object]]) -> str:
    step_blocks = []
    for event in events:
        step_blocks.append(
            "\n".join(
                [
                    f"### Step {event['step']}",
                    "",
                    f"- Action: `{json.dumps(event['action'])}`",
                    f"- Result: `{json.dumps(event['result'])}`",
                    f"- Immediate reward: `{event['reward']}`",
                    f"- Cumulative reward: `{event['cumulative_reward']}`",
                ]
            )
        )
    return "\n\n".join(
        [
            f"# Demo Transcript: {policy}",
            "",
            f"- Task type: `{task_type}`",
            f"- Seed: `{seed}`",
            f"- Final cumulative reward: `{final_reward}`",
            "",
            *step_blocks,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", default="coalition_market")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy", default="rule_based_expert", choices=sorted(POLICIES))
    parser.add_argument("--output", default="artifacts/demo_transcript.md")
    parser.add_argument("--search-seeds", type=int, default=0)
    args = parser.parse_args()

    best_seed = args.seed
    best_obs = None
    best_events: list[dict[str, object]] = []
    best_score = None

    seed_candidates = range(args.search_seeds) if args.search_seeds > 0 else [args.seed]
    for seed in seed_candidates:
        obs, events = run_episode(args.task_type, seed, args.policy)
        invalid_count = sum(1 for event in events if not event["result"].get("ok", True))
        score = (invalid_count, -float(obs.cumulative_reward))
        if best_score is None or score < best_score:
            best_score = score
            best_seed = seed
            best_obs = obs
            best_events = events

    assert best_obs is not None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_markdown(args.task_type, best_seed, args.policy, best_obs.cumulative_reward, best_events),
        encoding="utf-8",
    )
    print({"output": str(output), "seed": best_seed, "final_reward": best_obs.cumulative_reward})


if __name__ == "__main__":
    main()
