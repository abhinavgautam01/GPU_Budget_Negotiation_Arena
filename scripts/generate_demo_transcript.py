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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", default="coalition_market")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy", default="rule_based_expert", choices=sorted(POLICIES))
    parser.add_argument("--output", default="artifacts/demo_transcript.md")
    args = parser.parse_args()

    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type=args.task_type, seed=args.seed))
    steps: list[str] = []

    while not obs.done:
        action = POLICIES[args.policy](obs)
        obs = env.step(action)
        result = obs.last_action_result.model_dump() if obs.last_action_result else {}
        steps.append(
            "\n".join(
                [
                    f"### Step {obs.round_index}",
                    "",
                    f"- Action: `{json.dumps(action.model_dump(exclude_none=True))}`",
                    f"- Result: `{json.dumps(result)}`",
                    f"- Immediate reward: `{obs.reward}`",
                    f"- Cumulative reward: `{obs.cumulative_reward}`",
                ]
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n\n".join(
            [
                f"# Demo Transcript: {args.policy}",
                "",
                f"- Task type: `{args.task_type}`",
                f"- Seed: `{args.seed}`",
                f"- Final cumulative reward: `{obs.cumulative_reward}`",
                "",
                *steps,
            ]
        ),
        encoding="utf-8",
    )
    print({"output": str(output), "final_reward": obs.cumulative_reward})


if __name__ == "__main__":
    main()

