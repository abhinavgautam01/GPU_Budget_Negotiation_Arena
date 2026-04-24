from __future__ import annotations

"""Minimal training entrypoint stub.

This file is intentionally lightweight: it proves the reward loop can be called
from a trainer while the full Colab/TRL setup is added later.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpu_budget_arena.baselines import always_accept_policy
from gpu_budget_arena.env import GpuBudgetNegotiationEnv
from gpu_budget_arena.models import ResetConfig


def main() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="market_round", seed=11))
    rewards = []
    while not obs.done:
        obs = env.step(always_accept_policy(obs))
        rewards.append(obs.reward)
    print({"episodes": 1, "steps": len(rewards), "cumulative_reward": obs.cumulative_reward})


if __name__ == "__main__":
    main()
