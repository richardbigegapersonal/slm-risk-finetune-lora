import argparse, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from .prompts import build_prompt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--adapters", type=str, default="artifacts/lora-risk")
    ap.add_argument("--text", type=str, required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapters)
    base = AutoModelForCausalLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base, args.adapters)
    prompt = build_prompt(args.text)
    ids = tok(prompt, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=64, do_sample=False)
    s = tok.decode(out[0], skip_special_tokens=True)
    # Extract JSON at the end
    js = s.split("Assistant:")[-1].strip()
    print(js)

if __name__ == "__main__":
    main()
