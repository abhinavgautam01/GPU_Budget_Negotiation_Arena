from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_baselines import POLICIES, run_episode


TASK_TYPES = ["single_trade", "market_round", "coalition_market"]


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, dict[str, list[float]]] = {}
    shock_records: dict[str, int] = {}
    for record in records:
        task_type = str(record["task_type"])
        policy = str(record["policy"])
        grouped.setdefault(task_type, {}).setdefault(policy, []).append(float(record["episode_reward"]))
        if task_type == "coalition_market":
            shock_records[policy] = shock_records.get(policy, 0) + 1

    return {
        task_type: {
            policy: {
                "mean_reward": round(statistics.mean(scores), 4),
                "stdev_reward": round(statistics.pstdev(scores), 4),
                "episodes": len(scores),
                "seed_split": "holdout",
                "shock_episodes": shock_records.get(policy, 0) if task_type == "coalition_market" else 0,
            }
            for policy, scores in sorted(policies.items())
        }
        for task_type, policies in grouped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seed-offset", type=int, default=50_000)
    parser.add_argument("--output", default="artifacts/holdout_eval.json")
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    for seed in range(args.seed_offset, args.seed_offset + args.seeds):
        for task_type in TASK_TYPES:
            for policy_name in POLICIES:
                records.append(run_episode(task_type, seed, policy_name))

    payload = {
        "protocol": {
            "split": "holdout",
            "seed_offset": args.seed_offset,
            "seeds": args.seeds,
            "task_types": TASK_TYPES,
            "note": "Holdout seeds are disjoint from training/demo seeds and exercise the same hidden-job, opponent, and shock generators.",
        },
        "summary": summarize(records),
        "episodes": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print({"output": str(output), "episodes": len(records), "seed_offset": args.seed_offset})


if __name__ == "__main__":
    main()
