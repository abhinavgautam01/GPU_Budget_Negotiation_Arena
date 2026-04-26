"""Tests for the LLM-as-policy adapter (CPU-only, no torch needed).

These tests exercise the parsing and prompt-rendering logic without ever
touching `transformers` or `torch`. The actual `make_llm_policy` factory is
covered indirectly by `scripts/evaluate_trained_llm.py` on Colab.
"""

from __future__ import annotations

from gpu_budget_arena.env import GpuBudgetNegotiationEnv
from gpu_budget_arena.llm_policy import (
    SYSTEM_PROMPT,
    parse_action_text,
    render_messages,
    render_user_prompt,
)
from gpu_budget_arena.models import ResetConfig


def test_render_messages_uses_system_prompt() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="single_trade", seed=0))
    messages = render_messages(obs)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "Observation:" in messages[1]["content"]
    assert obs.task_id in messages[1]["content"]


def test_render_user_prompt_includes_observation_fields() -> None:
    env = GpuBudgetNegotiationEnv()
    obs = env.reset(ResetConfig(task_type="market_round", seed=1))
    prompt = render_user_prompt(obs)
    assert prompt.startswith("Given the current GPU market observation")
    assert "controlled_lab_budget" in prompt
    assert "controlled_lab_reputation" in prompt
    assert prompt.rstrip().endswith("Return only one JSON action object.")


def test_parse_action_text_plain_json_succeeds() -> None:
    action, raw = parse_action_text('{"action_type":"wait"}')
    assert action is not None
    assert action.action_type == "wait"
    assert raw == '{"action_type":"wait"}'


def test_parse_action_text_strips_markdown_fence() -> None:
    text = (
        "Sure, here is the action:\n"
        "```json\n"
        '{"action_type":"send_offer","target_lab_id":"lab_1",'
        '"block_ids":["b_0_0"],"requested_block_ids":["b_1_0"],'
        '"payment":2.0,"message":"swap"}\n'
        "```"
    )
    action, _ = parse_action_text(text)
    assert action is not None
    assert action.action_type == "send_offer"
    assert action.target_lab_id == "lab_1"
    assert action.block_ids == ["b_0_0"]


def test_parse_action_text_returns_none_on_garbage() -> None:
    action, raw = parse_action_text("totally not json")
    assert action is None
    assert "totally not json" in raw


def test_parse_action_text_handles_extra_prose_around_json() -> None:
    text = (
        "I think the right move is\n"
        '{"action_type":"finish"}\n'
        "and we are done."
    )
    action, _ = parse_action_text(text)
    assert action is not None
    assert action.action_type == "finish"
