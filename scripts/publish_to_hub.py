"""
Publish the SFT (or GRPO) LoRA adapter to a Hugging Face model repo.

Usage (Colab or local with HF_TOKEN env var set):

    python3 scripts/publish_to_hub.py \\
        --checkpoint artifacts/sft-checkpoint \\
        --repo abhinavgautam01/gpu-budget-arena-llama-3.2-3b-sft \\
        --base-model unsloth/Llama-3.2-3B-Instruct \\
        --stage sft

This uploads:

  - The LoRA adapter files (adapter_config.json, adapter_model.safetensors, tokenizer.*)
  - A model card (README.md) describing the env, training pipeline, and headline
    eval numbers, with code snippets a judge can copy-paste.

Why ship this script:
  * Other submissions (e.g. ashutosh111/negotiation-vendor-llama32-1b-grpo-step25)
    publish their adapter to HF Hub so judges can pull weights with one line of
    code. We want the same affordance.
  * Without this, our 80 MB+ checkpoint can only live on Drive — Hugging Face
    Spaces explicitly rejects raw binaries, and a reviewer who clones the repo
    won't see the model.

The script is read-only against the repo (only writes to /tmp); it never
mutates the local checkpoint and is safe to re-run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_summary(stage: str) -> dict:
    art = ROOT / "artifacts"
    out: dict = {}
    if (art / "trained_llm_summary.json").exists():
        out["trained_llm_summary"] = json.loads(
            (art / "trained_llm_summary.json").read_text(encoding="utf-8")
        )
    if (art / "sft_training_curve.json").exists():
        out["sft_curve_summary"] = json.loads(
            (art / "sft_training_curve.json").read_text(encoding="utf-8")
        ).get("summary", {})
    if (art / "grpo_training_curve.json").exists():
        out["grpo_curve_summary"] = json.loads(
            (art / "grpo_training_curve.json").read_text(encoding="utf-8")
        ).get("summary", {})
    out["stage"] = stage
    return out


def _render_model_card(repo: str, base_model: str, stage: str, summary: dict) -> str:
    sft = summary.get("sft_curve_summary", {})
    grpo = summary.get("grpo_curve_summary", {})
    rows = (summary.get("trained_llm_summary") or {}).get("rows", [])
    sft_row = next(
        (r for r in rows if r.get("policy") == "trained_llm_sft"), None
    )
    base_row = next(
        (r for r in rows if r.get("policy") == "base_instruct_llm"), None
    )

    lines: list[str] = []
    lines.append("---")
    lines.append("library_name: peft")
    lines.append(f"base_model: {base_model}")
    lines.append("tags:")
    lines.append("- llama")
    lines.append("- lora")
    lines.append("- unsloth")
    lines.append("- trl")
    lines.append("- grpo")
    lines.append("- negotiation")
    lines.append("- multi-agent")
    lines.append("- openenv")
    lines.append("license: llama3.2")
    lines.append("---")
    lines.append("")
    lines.append(f"# GPU Budget Negotiation Arena · Llama-3.2-3B · Stage = `{stage}`")
    lines.append("")
    lines.append(
        "LoRA adapter trained on the GPU Budget Negotiation Arena multi-agent "
        "OpenEnv. Five fictional AI labs share a single GPU pool just before a "
        "paper deadline, with hidden private utility, deadlines, budgets, "
        "reputation, and supply shocks. The trainable lab (`lab_0`) sends "
        "offers, counter-offers, reservations, allocations, coalition "
        "commitments, and natural-language pitches."
    )
    lines.append("")
    lines.append("- Live Space: <https://abhinavgautam01-gpu-budget-negotiation-arena.hf.space>")
    lines.append("- Source: <https://github.com/abhinavgautam01/GPU_Budget_Negotiation_Arena>")
    lines.append("- 3-minute pitch: [`PITCH.md`](https://github.com/abhinavgautam01/GPU_Budget_Negotiation_Arena/blob/main/PITCH.md)")
    lines.append("- Judge Q&A bank: [`JUDGE_QA.md`](https://github.com/abhinavgautam01/GPU_Budget_Negotiation_Arena/blob/main/JUDGE_QA.md)")
    lines.append("")
    lines.append("## Training pipeline")
    lines.append("")
    lines.append("**Stage 1 — Unsloth SFT.** Base `unsloth/Llama-3.2-3B-Instruct` was fine-tuned on chat-formatted expert-replay traces from the env (`data/sft_messages.jsonl`).")
    if sft:
        lines.append("")
        lines.append(f"- Steps: `{sft.get('total_steps')}` over `{sft.get('num_train_epochs')}` epochs")
        lines.append(f"- First loss: `{sft.get('first_loss')}` → final loss: `{sft.get('final_loss')}` ({sft.get('loss_drop_pct')}% drop)")
    lines.append("")
    lines.append("**Stage 2 — TRL GRPO against the live env reward.** The SFT'd LoRA was warm-started into `trl.GRPOTrainer` with a custom reward function that calls `GpuBudgetNegotiationEnv.step(action).reward + format_bonus` on each sampled completion (parse penalty if JSON is malformed). No surrogate, no learned reward model.")
    if grpo:
        lines.append("")
        lines.append(f"- Steps: `{grpo.get('max_steps')}` × `{grpo.get('num_generations')}` completions / step on `{grpo.get('prompts')}` replayed env observations")
        lines.append(f"- First step mean reward: `{grpo.get('first_step_mean_reward')}` → last step: `{grpo.get('last_step_mean_reward')}` (peak `{grpo.get('best_step_mean_reward')}`)")
    lines.append("")
    lines.append("## Headline eval results")
    lines.append("")
    lines.append("Live policy rollouts on 5 seeds × 3 tasks against scripted baselines.")
    lines.append("")
    if sft_row and base_row:
        lines.append("| Task | Base Llama (untrained) | This adapter (SFT) | Δ |")
        lines.append("|---|---:|---:|---:|")
        for t in ("single_trade", "market_round", "coalition_market"):
            b = base_row.get(f"{t}_mean")
            s = sft_row.get(f"{t}_mean")
            if isinstance(b, (int, float)) and isinstance(s, (int, float)):
                lines.append(f"| `{t}` | `{b:+.4f}` | `{s:+.4f}` | `{s - b:+.4f}` |")
        if isinstance(base_row.get("overall_mean"), (int, float)) and isinstance(sft_row.get("overall_mean"), (int, float)):
            lines.append(f"| **overall** | `{base_row['overall_mean']:+.4f}` | `{sft_row['overall_mean']:+.4f}` | `{sft_row['overall_mean'] - base_row['overall_mean']:+.4f}` |")
        lines.append("")
        lines.append("Sign flips on `market_round` and `coalition_market` versus the same untrained Llama — third out of eight evaluated policies, ahead of every scripted bot except the always-accept opportunist and the hand-authored rule-expert ceiling.")
    lines.append("")
    lines.append("## Quick load")
    lines.append("")
    lines.append("```python")
    lines.append("from peft import PeftModel")
    lines.append("from transformers import AutoModelForCausalLM, AutoTokenizer")
    lines.append("")
    lines.append(f"base_model_id = \"{base_model}\"")
    lines.append(f"adapter_id    = \"{repo}\"")
    lines.append("")
    lines.append("tok   = AutoTokenizer.from_pretrained(base_model_id)")
    lines.append("model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map=\"auto\", load_in_4bit=True)")
    lines.append("model = PeftModel.from_pretrained(model, adapter_id)")
    lines.append("model.eval()")
    lines.append("```")
    lines.append("")
    lines.append("Or roll it out as `lab_0` against every scripted baseline:")
    lines.append("")
    lines.append("```bash")
    lines.append("git clone https://github.com/abhinavgautam01/GPU_Budget_Negotiation_Arena.git")
    lines.append("cd GPU_Budget_Negotiation_Arena")
    lines.append("python3 scripts/evaluate_trained_llm.py \\")
    lines.append(f"  --base-model {base_model} \\")
    lines.append(f"  --model-path {repo} \\")
    lines.append("  --policy-name trained_llm_sft --include-base-model --seeds 5 \\")
    lines.append("  --output artifacts/trained_llm_eval.json")
    lines.append("```")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Always-accept gets `0.266` overall vs this model's `0.257` — within noise. We match the structural greedy ceiling on a single scalar; we beat it on `coalition_market` (`+0.42` vs `+0.23`), under the judge layer (always-accept generates no pitch), and on coalition reliability (always-accept breaches contracted commitments under capacity shocks).")
    lines.append("- Parse-failure rate at this checkpoint is ~50%. Non-parseable completions fall back to the no-op action; episodes don't crash. GRPO continues to push this down (the GRPO reward curve includes a format bonus).")
    lines.append("- Llama-3.2-3B was chosen to fit a free Colab T4. The pipeline is backbone-agnostic — `--base-model unsloth/Llama-3.1-8B-Instruct` works unchanged on a paid runtime.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="Local path to LoRA checkpoint (e.g. artifacts/sft-checkpoint)")
    ap.add_argument("--repo", required=True, help="HF Hub repo id, e.g. user/gpu-budget-arena-llama-3.2-3b-sft")
    ap.add_argument("--base-model", default="unsloth/Llama-3.2-3B-Instruct")
    ap.add_argument("--stage", choices=["sft", "grpo", "sft+grpo"], default="sft")
    ap.add_argument("--private", action="store_true", help="Create the repo as private")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (defaults to $HF_TOKEN)")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    if not args.token:
        raise SystemExit("set --token or export HF_TOKEN before running")

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise SystemExit("pip install huggingface_hub  # required for hub upload")

    api = HfApi(token=args.token)
    create_repo(args.repo, token=args.token, private=args.private, exist_ok=True, repo_type="model")
    print(json.dumps({"status": "repo-ready", "repo": args.repo}))

    summary = _load_summary(args.stage)
    card = _render_model_card(args.repo, args.base_model, args.stage, summary)
    card_path = Path("/tmp") / "MODEL_CARD.md"
    card_path.write_text(card, encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        token=args.token,
        commit_message=f"Model card for {args.stage} checkpoint",
    )
    print(json.dumps({"status": "card-uploaded", "bytes": len(card)}))

    api.upload_folder(
        folder_path=str(ckpt),
        repo_id=args.repo,
        repo_type="model",
        token=args.token,
        commit_message=f"{args.stage} checkpoint upload",
        ignore_patterns=["*.bin.index*", "*.tmp", "checkpoint-*/optimizer.pt"],
    )
    print(json.dumps({"status": "ok", "repo_url": f"https://huggingface.co/{args.repo}"}))


if __name__ == "__main__":
    main()
