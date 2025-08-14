import argparse, json, re, ast
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from .data import read_jsonl
from .prompts import build_prompt

LABELS = {"FRAUD","MERCHANT_ERROR","PASSWORD_RESET","INFO"}

def parse_json(s: str) -> Dict[str, Any]:
    # robust JSON extractor for small outputs
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m: return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        try:
            return ast.literal_eval(m.group(0))
        except Exception:
            return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--adapters", type=str, default="artifacts/lora-risk")
    ap.add_argument("--val", type=str, default="data/risk_val.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapters)
    base = AutoModelForCausalLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base, args.adapters)

    data = read_jsonl(args.val)
    total = 0; correct = 0; valid_json = 0
    for item in data:
        prompt = build_prompt(item["input"])
        ids = tok(prompt, return_tensors="pt").input_ids
        out = model.generate(ids, max_new_tokens=64, do_sample=False)
        s = tok.decode(out[0], skip_special_tokens=True)
        js = parse_json(s)
        total += 1
        if isinstance(js, dict) and "label" in js and js["label"] in LABELS:
            valid_json += 1
            gold = json.loads(item["output"])["label"]
            if js["label"] == gold: correct += 1

    metrics = {"n": total, "json_valid": valid_json/total, "accuracy": correct/total}
    print(json.dumps(metrics, indent=2))
    with open("artifacts/metrics.json","w") as f: json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
