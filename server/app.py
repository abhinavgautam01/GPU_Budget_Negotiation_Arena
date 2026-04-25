from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from gpu_budget_arena.env import GpuBudgetNegotiationEnv
from gpu_budget_arena.models import GpuNegotiationAction, ResetConfig

app = FastAPI(title="GPU Budget Negotiation Arena", version="0.1.0")
env = GpuBudgetNegotiationEnv()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "benchmark_id": "gpu_budget_negotiation"}


@app.get("/tasks")
def tasks() -> dict[str, object]:
    return {
        "benchmark_id": "gpu_budget_negotiation",
        "tasks": [
            {"task_type": "single_trade", "difficulty": "easy"},
            {"task_type": "market_round", "difficulty": "medium"},
            {"task_type": "coalition_market", "difficulty": "hard"},
        ],
        "features": [
            "coalitions",
            "adaptive_bot_pitches",
            "optional_rule_judge",
            "holdout_seed_evaluation",
            "redacted_public_state",
            "dynamic_market_shocks",
        ],
    }


@app.post("/reset")
def reset(config: ResetConfig) -> dict[str, object]:
    return {"observation": env.reset(config).model_dump(mode="json")}


@app.post("/step")
def step(action: GpuNegotiationAction) -> dict[str, object]:
    return {"observation": env.step(action).model_dump(mode="json")}


@app.get("/state")
def state(include_private: bool = False) -> dict[str, object]:
    if include_private:
        if os.getenv("GPU_ARENA_DEBUG_STATE") != "1":
            raise HTTPException(
                status_code=403,
                detail="Private state is disabled. Set GPU_ARENA_DEBUG_STATE=1 for local debugging.",
            )
        return {"state": env.state()}
    return {"state": env.public_state()}
