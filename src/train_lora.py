import os, json, math, argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .data import build_supervised_dataset, Collator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default=os.environ.get("BASE_MODEL","TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    ap.add_argument("--train", type=str, default="data/risk_train.jsonl")
    ap.add_argument("--val", type=str, default="data/risk_val.jsonl")
    ap.add_argument("--out", type=str, default="artifacts/lora-risk")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--grad_acc", type=int, default=4)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tok.pad_token = tok.eos_token

    train_data = build_supervised_dataset(args.train, tok)
    val_data   = build_supervised_dataset(args.val, tok)

    collator = Collator(tok)

    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)

    def tokenize_fn(samples):
        return samples  # collator handles it

    args_tr = TrainingArguments(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_acc,
        num_train_epochs=args.epochs,
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        bf16=False,
        fp16=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    with open(Path(args.out)/"training_complete.json","w") as f:
        json.dump({"ok": True}, f)

if __name__ == "__main__":
    main()
