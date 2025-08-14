import json, random
from typing import List, Dict
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase

def read_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items

@dataclass
class Collator:
    tokenizer: PreTrainedTokenizerBase
    def __call__(self, batch):
        texts = [b["prompt"] for b in batch]
        labels = [b["target"] for b in batch]
        enc = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            lab = self.tokenizer(labels, truncation=True, padding=True, return_tensors="pt")
        enc["labels"] = lab["input_ids"]
        return enc

def build_supervised_dataset(jsonl_path: str, tokenizer: PreTrainedTokenizerBase):
    from .prompts import build_prompt
    raw = read_jsonl(jsonl_path)
    data = []
    for r in raw:
        prompt = build_prompt(r["input"])  # prepend system + user
        data.append({"prompt": prompt, "target": r["output"]})
    return data
