from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab-safe optional Unsloth SFT warm start.")
    parser.add_argument("--model-name", default="unsloth/Llama-3.2-3B-Instruct")
    parser.add_argument("--data", default="data/sft_messages.jsonl")
    parser.add_argument("--output", default="artifacts/sft-checkpoint")
    parser.add_argument("--drive-output", default="")
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--push-to-hub", default="", help="Optional model repo id for uploading the trained LoRA adapter.")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print({"status": "skipped", "reason": "PyTorch is not installed."})
        return

    if not torch.cuda.is_available():
        print(
            {
                "status": "skipped",
                "reason": "Unsloth requires a CUDA GPU. In Colab select Runtime -> Change runtime type -> T4 GPU, then rerun from the setup cell.",
                "torch": getattr(torch, "__version__", "unknown"),
            }
        )
        return

    # Unsloth must be imported before TRL/transformers/peft for its patches.
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run scripts/check_submission.py or build_sft_dataset.py first.")

    raw_dataset = load_dataset("json", data_files=str(data_path), split="train")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    def render_text(example: dict[str, object]) -> dict[str, str]:
        messages = example["messages"]
        if not isinstance(messages, list):
            raise ValueError("Expected each SFT row to contain a messages list.")
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if not isinstance(text, str):
            raise TypeError("chat template must render to a string")
        return {"text": text}

    dataset = raw_dataset.map(render_text, remove_columns=raw_dataset.column_names)

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": SFTConfig(
            output_dir=args.output,
            max_steps=args.max_steps,
            dataset_text_field="text",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=10,
            save_steps=max(60, args.max_steps // 2),
            report_to=[],
            packing=False,
        ),
    }
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    resume_from_checkpoint = args.resume_from_checkpoint or None
    if resume_from_checkpoint and not Path(resume_from_checkpoint).exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(args.output)

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)

    if args.drive_output:
        drive_output = Path(args.drive_output)
        drive_output.mkdir(parents=True, exist_ok=True)
        target = drive_output / Path(args.output).name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(args.output, target)
        print({"status": "ok", "output": args.output, "drive_output": str(target), "resumed_from": resume_from_checkpoint})
    else:
        print({"status": "ok", "output": args.output, "resumed_from": resume_from_checkpoint})


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        print({"status": "skipped", "reason": str(exc)})
        sys.exit(0)
