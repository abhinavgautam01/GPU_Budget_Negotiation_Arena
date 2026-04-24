from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")


def run(cmd: list[str]) -> None:
    print({"running": " ".join(cmd)})
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    artifacts_dir = ROOT / "artifacts"
    plots_dir = ROOT / "plots"
    artifacts_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    run([sys.executable, "-m", "pytest", "-q"])
    run([sys.executable, "scripts/smoke.py"])
    run([sys.executable, "scripts/evaluate_baselines.py", "--seeds", "10", "--output", "artifacts/baseline_eval.json"])
    run([sys.executable, "scripts/plot_eval.py", "--input", "artifacts/baseline_eval.json", "--output", "plots/baseline_rewards.svg"])
    run([sys.executable, "scripts/generate_demo_transcript.py", "--task-type", "coalition_market", "--seed", "7", "--policy", "rule_based_expert", "--output", "artifacts/demo_transcript.md"])

    payload = json.loads((artifacts_dir / "baseline_eval.json").read_text(encoding="utf-8"))
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
