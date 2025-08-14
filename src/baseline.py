import argparse, json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def parse(jsonl_path):
    X=[]; y=[]
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            X.append(obj["input"])
            y.append(json.loads(obj["output"])['label'])
    return X,y

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default="data/risk_train.jsonl")
    ap.add_argument("--val", type=str, default="data/risk_val.jsonl")
    args = ap.parse_args()
    Xtr, ytr = parse(args.train); Xv, yv = parse(args.val)
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(Xtr, ytr)
    yp = pipe.predict(Xv)
    acc = accuracy_score(yv, yp)
    print(json.dumps({"baseline_accuracy": acc}, indent=2))
