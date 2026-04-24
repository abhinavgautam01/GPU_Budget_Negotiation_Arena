from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are negotiating for scarce GPU capacity in a multi-agent market. "
    "Respond with one valid JSON action object only."
)


def build_prompt(observation: dict[str, object]) -> str:
    compact = {
        "task_id": observation["task_id"],
        "difficulty": observation["difficulty"],
        "round_index": observation["round_index"],
        "controlled_lab_budget": observation["controlled_lab_budget"],
        "controlled_lab_reputation": observation["controlled_lab_reputation"],
        "private_jobs": observation["private_jobs"],
        "owned_blocks": observation["owned_blocks"],
        "visible_labs": observation["visible_labs"],
        "active_offers": observation["active_offers"],
        "active_coalitions": observation["active_coalitions"],
        "last_action_result": observation["last_action_result"],
    }
    return (
        "Given the current GPU market observation, choose the next action.\n\n"
        f"Observation:\n{json.dumps(compact, separators=(',', ':'))}\n\n"
        "Return only one JSON action object."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sft_traces.jsonl")
    parser.add_argument("--output", default="data/sft_messages.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            trace = json.loads(line)
            action = json.dumps(trace["action"], separators=(",", ":"))
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(trace["observation"])},
                    {"role": "assistant", "content": action},
                ],
                "task_type": trace["task_type"],
                "seed": trace["seed"],
                "round_index": trace["round_index"],
            }
            dst.write(json.dumps(record) + "\n")
            rows += 1

    print({"input": str(input_path), "output": str(output_path), "rows": rows})


if __name__ == "__main__":
    main()

