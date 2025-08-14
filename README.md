# SLM Fine-Tuning with LoRA for Risk Scoring (Hands-on)

A compact project showing how to fine-tune a **Small Language Model** (SLM) via **LoRA** to classify customer messages into risk triage labels:
**FRAUD, MERCHANT_ERROR, PASSWORD_RESET, INFO**. Includes **evaluation, inference, and a LogReg baseline**.

> GPU recommended for LoRA. If you don't have one, run the baseline first.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option A — Baseline (no GPU needed)
```bash
python src/baseline.py
```

### Option B — LoRA fine-tune (GPU recommended)
Pick a small base model; default is TinyLlama:
```bash
# ENV override optional:
# export BASE_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
python -m src.train_lora --epochs 1 --bsz 4 --grad_acc 4
```

Evaluate on validation set:
```bash
python -m src.eval
# -> prints { "n": ..., "json_valid": ..., "accuracy": ... } and writes artifacts/metrics.json
```

Run inference:
```bash
python -m src.infer --text "Charged twice for the same purchase."
# -> { "label": "MERCHANT_ERROR" }
```

## How it works
- **Data format**: `data/risk_*.jsonl` with `instruction`, `input`, `output` (JSON with `"label"`).
- **Prompts**: `src/prompts.py` prepends a strict **system prompt** requiring JSON-only output.
- **Training**: `src/train_lora.py` applies **LoRA adapters** (`q_proj, k_proj, v_proj, o_proj`) on a causal LM.
- **Evaluation**: `src/eval.py` checks **JSON validity** and **accuracy** against the gold label.
- **Baseline**: `src/baseline.py` runs a TF-IDF + Logistic Regression pipeline for a quick, non-neural baseline.

## Files
```
slm-risk-finetune-lora/
├─ data/
│  ├─ risk_train.jsonl        # synthetic train set
│  └─ risk_val.jsonl          # synthetic validation set
├─ src/
│  ├─ prompts.py              # system prompt + build_prompt()
│  ├─ data.py                 # JSONL reader + collator
│  ├─ train_lora.py           # LoRA fine-tuning (Transformers + PEFT)
│  ├─ eval.py                 # JSON validity + accuracy metrics
│  ├─ infer.py                # load adapters and classify new text
│  └─ baseline.py             # TF-IDF + LogisticRegression baseline
├─ artifacts/                 # model adapters + metrics.json
├─ tests/                     # (slot for future unit tests)
└─ requirements.txt
```

## Emphasize:
- **Why SLM + LoRA?** Fast, cheap, controllable; ideal for structured tasks (classification/extraction) vs. general chat.
- **Output contracts:** JSON schema enforces **predictable, auditable** outputs (tie to compliance).
- **Ground truth & eval:** accuracy + **JSON validity rate**; can add **per-class recall** (e.g., prioritize FRAUD recall).
- **Fallbacks & safety:** If LoRA fails, **baseline** or **rule-based** classifier can keep SLAs.
- **Governance**: version prompts, adapters, metrics in `artifacts/`; attach to a **model registry**.

## Next steps
- Add **class weights** or focal loss for real-world imbalance.
- Swap to an **instruction-tuned SLM** (e.g., TinyLlama, Qwen 0.5–1.8B) for stronger zero-shot.
- Integrate with your **MLOps gates** (see Chapter 8): eval thresholds → staging → canary.
