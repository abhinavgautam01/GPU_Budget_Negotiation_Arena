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

    dataset = load_dataset("json", data_files=str(data_path), split="train")
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

    def formatting_func(example: dict[str, object]) -> list[str]:
        messages = example["messages"]
        if not isinstance(messages, list):
            raise ValueError("Expected each SFT row to contain a messages list.")
        return [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)]

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "formatting_func": formatting_func,
        "args": SFTConfig(
            output_dir=args.output,
            max_steps=args.max_steps,
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

    trainer.train()
    trainer.save_model(args.output)

    if args.drive_output:
        drive_output = Path(args.drive_output)
        drive_output.mkdir(parents=True, exist_ok=True)
        target = drive_output / Path(args.output).name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(args.output, target)
        print({"status": "ok", "output": args.output, "drive_output": str(target)})
    else:
        print({"status": "ok", "output": args.output})


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        print({"status": "skipped", "reason": str(exc)})
        sys.exit(0)
