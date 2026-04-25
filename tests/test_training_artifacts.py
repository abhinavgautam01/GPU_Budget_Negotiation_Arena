from __future__ import annotations

import json
from pathlib import Path


def test_training_curve_records_reward_improvement() -> None:
    curve_path = Path("artifacts/training_curve.json")
    assert curve_path.exists()
    curve = json.loads(curve_path.read_text(encoding="utf-8"))
    assert len(curve) >= 3
    assert curve[-1]["eval_mean_reward"] > curve[0]["eval_mean_reward"]
    assert "selected_policies" in curve[-1]


def test_before_after_transcript_records_behavior_change() -> None:
    transcript = Path("artifacts/before_after_training.md").read_text(encoding="utf-8")
    assert "Before Training" in transcript
    assert "After Training" in transcript
    assert "form_coalition" in transcript or "allocate_to_job" in transcript
