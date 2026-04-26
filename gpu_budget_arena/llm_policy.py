"""LLM-as-policy adapter for the GPU Budget Negotiation Arena.

Wraps a Hugging Face causal LM + tokenizer pair as a policy callable that
matches the existing scripted-baseline signature::

    Callable[[GpuNegotiationObservation], GpuNegotiationAction]

The prompt format is identical to the SFT chat data (`scripts/build_sft_dataset.py`)
so an SFT'd or GRPO'd checkpoint can negotiate on the live environment without
any additional templating. On every step the adapter:

  1. Renders the observation as a compact JSON inside a chat template.
  2. Samples a completion from the model.
  3. Validates the completion with `GpuNegotiationAction.model_validate_json`.
  4. Falls back to `wait` if the model produced malformed output, so the
     environment never sees an unparseable action and the rollout always
     completes cleanly.

This module imports torch / transformers lazily inside the factory so that
`from gpu_budget_arena.llm_policy import ...` is safe on a CPU-only machine
that does not have those packages installed at all.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from gpu_budget_arena.models import (
    GpuNegotiationAction,
    GpuNegotiationObservation,
)


SYSTEM_PROMPT = (
    "You are negotiating for scarce GPU capacity in a multi-agent market. "
    "Respond with one valid JSON action object only."
)


def _compact_observation(obs: GpuNegotiationObservation) -> dict[str, Any]:
    """Same projection used during SFT data generation."""
    payload = obs.model_dump(mode="json")
    return {
        "task_id": payload["task_id"],
        "difficulty": payload["difficulty"],
        "round_index": payload["round_index"],
        "controlled_lab_budget": payload["controlled_lab_budget"],
        "controlled_lab_reputation": payload["controlled_lab_reputation"],
        "private_jobs": payload["private_jobs"],
        "owned_blocks": payload["owned_blocks"],
        "visible_labs": payload["visible_labs"],
        "active_offers": payload["active_offers"],
        "active_coalitions": payload["active_coalitions"],
        "last_action_result": payload["last_action_result"],
    }


def render_user_prompt(obs: GpuNegotiationObservation) -> str:
    obs_json = json.dumps(_compact_observation(obs), separators=(",", ":"))
    return (
        "Given the current GPU market observation, choose the next action.\n\n"
        f"Observation:\n{obs_json}\n\n"
        "Return only one JSON action object."
    )


def render_messages(obs: GpuNegotiationObservation) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(obs)},
    ]


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_first_json_object(text: str) -> str | None:
    """Return the first balanced { ... } substring in `text`, or None.

    Models occasionally pad the JSON with chat tokens, code fences, or
    natural-language commentary. We strip those by finding the first balanced
    object so that downstream `model_validate_json` succeeds whenever the
    model's intent is recoverable.
    """
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    candidate = match.group(0)
    depth = 0
    end = -1
    for idx, ch in enumerate(candidate):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end == -1:
        return None
    return candidate[:end]


def parse_action_text(text: str) -> tuple[GpuNegotiationAction | None, str]:
    """Try to parse model output as an action; return (action_or_none, raw_json)."""
    candidate = _extract_first_json_object(text) or text.strip()
    if not candidate:
        return None, ""
    try:
        action = GpuNegotiationAction.model_validate_json(candidate)
        return action, candidate
    except Exception:
        return None, candidate


@dataclass
class LlmPolicyConfig:
    max_new_tokens: int = 192
    temperature: float = 0.6
    top_p: float = 0.9
    do_sample: bool = True
    fallback_action_type: str = "wait"


def make_llm_policy(
    model: Any,
    tokenizer: Any,
    config: LlmPolicyConfig | None = None,
) -> Callable[[GpuNegotiationObservation], GpuNegotiationAction]:
    """Return a policy callable that uses `model` + `tokenizer` for action generation.

    The callable matches the scripted-policy signature used everywhere else
    in the repo, so it can drop straight into `evaluate_baselines.py`,
    `train_grpo_stub.py`, or any other rollout loop.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        A causal-LM ready for `.generate(...)`. Should already be on the
        right device and in eval mode.
    tokenizer : transformers.PreTrainedTokenizerBase
        Must support `apply_chat_template`. Setting `pad_token` is handled
        here if the tokenizer left it empty.
    config : LlmPolicyConfig, optional
        Sampling overrides.
    """

    cfg = config or LlmPolicyConfig()

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = "cpu"

    def _policy(obs: GpuNegotiationObservation) -> GpuNegotiationAction:
        messages = render_messages(obs)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        try:
            import torch  # local import keeps top-level CPU-safe

            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    do_sample=cfg.do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        except Exception:
            return GpuNegotiationAction(action_type=cfg.fallback_action_type)

        prompt_len = inputs["input_ids"].shape[1]
        completion = tokenizer.decode(
            output[0][prompt_len:], skip_special_tokens=True
        )
        action, _ = parse_action_text(completion)
        if action is None:
            return GpuNegotiationAction(action_type=cfg.fallback_action_type)
        return action

    return _policy


__all__ = [
    "LlmPolicyConfig",
    "SYSTEM_PROMPT",
    "make_llm_policy",
    "parse_action_text",
    "render_messages",
    "render_user_prompt",
]
