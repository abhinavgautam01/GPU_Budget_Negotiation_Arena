from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpu_budget_arena.baselines import rule_based_expert_policy
from gpu_budget_arena.env import GpuBudgetNegotiationEnv
from gpu_budget_arena.models import ResetConfig


def generate_trace(task_type: str, seed: int) -> list[dict[str, object]]:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type=task_type, seed=seed))
    rows: list[dict[str, object]] = []
    while not obs.done:
        action = rule_based_expert_policy(obs)
        rows.append(
            {
                "task_type": task_type,
                "seed": seed,
                "round_index": obs.round_index,
                "observation": obs.model_dump(mode="json"),
                "action": action.model_dump(mode="json", exclude_none=True),
            }
        )
        obs = env.step(action)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/sft_traces.jsonl")
    parser.add_argument("--seeds", type=int, default=25)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    task_types = ["single_trade", "market_round", "coalition_market"]
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for seed in range(args.seeds):
            for task_type in task_types:
                for row in generate_trace(task_type, seed):
                    handle.write(json.dumps(row) + "\n")
                    count += 1
    print({"output": str(output), "rows": count})


if __name__ == "__main__":
    main()
