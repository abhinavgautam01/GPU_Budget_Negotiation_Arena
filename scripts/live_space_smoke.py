from __future__ import annotations

import argparse
import json
import urllib.request
from urllib.error import URLError


def get_json(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="https://abhinavgautam01-gpu-budget-negotiation-arena.hf.space",
    )
    args = parser.parse_args()

    try:
        health = get_json(f"{args.base_url}/health")
        tasks = get_json(f"{args.base_url}/tasks")
        reset = post_json(f"{args.base_url}/reset", {"task_type": "market_round", "seed": 42})
    except URLError as exc:
        print(
            json.dumps(
                {
                    "base_url": args.base_url,
                    "ok": False,
                    "error": str(exc),
                    "note": "Network or DNS resolution failed from this machine. Re-run on a machine with internet access.",
                },
                indent=2,
            )
        )
        return

    observation = reset["observation"]
    print(json.dumps(
        {
            "base_url": args.base_url,
            "ok": True,
            "health": health,
            "task_types": [task["task_type"] for task in tasks["tasks"]],
            "reset_task_id": observation["task_id"],
            "visible_labs": len(observation["visible_labs"]),
            "active_offers": len(observation["active_offers"]),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
