"""Download the public datasets used for training and evaluation.

Datasets
--------
1. TweetEval sentiment (Barbieri et al., 2020; SemEval-2017 Task 4A).
   3-class (negative / neutral / positive), informal social-media text.
   Official train / validation / test splits are used as-is.
2. Lenovo K8 Note Amazon.in reviews (~14.7k, binary positive / negative).
   Informal Indian-English product reviews. Split 70 / 10 / 20 (stratified, seed 42).
3. Amazon memory-card reviews (~4.9k, 1-5 stars). Used ONLY as a held-out,
   out-of-domain test set. Stars mapped to 1-2 = negative, 3 = neutral, 4-5 = positive.

Run:  python scripts/download_data.py
"""
import io
import os
import zipfile

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

RAW = "data/raw"
OUT = "data/processed"
TWEETEVAL = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/{split}_{kind}.txt"
K8_ZIP = ("https://raw.githubusercontent.com/kameshwaran1995/Topic-Analysis-of-Review-Data/main/"
          "topic%20analysis%20of%20review%20data.zip")
AMAZON = ("https://raw.githubusercontent.com/AnayDubey/Amazon-Product-Reviews-Sentiment-Analysis/main/"
          "amazon_reviews.csv")
TWEETEVAL_LABELS = {0: "negative", 1: "neutral", 2: "positive"}


def fetch(url):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def tweeteval():
    for split, name in [("train", "train"), ("val", "val"), ("test", "test")]:
        text = fetch(TWEETEVAL.format(split=split, kind="text")).decode("utf-8").split("\n")
        labels = fetch(TWEETEVAL.format(split=split, kind="labels")).decode("utf-8").split("\n")
        rows = [(t.strip(), TWEETEVAL_LABELS[int(l)]) for t, l in zip(text, labels) if l.strip()]
        pd.DataFrame(rows, columns=["text", "label"]).to_csv(f"{OUT}/tweeteval_{name}.csv", index=False)


def k8():
    z = zipfile.ZipFile(io.BytesIO(fetch(K8_ZIP)))
    name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    df = pd.read_csv(z.open(name)).dropna()
    df = pd.DataFrame({"text": df["review"].astype(str).str.strip(),
                       "label": df["sentiment"].map({1: "positive", 0: "negative"})})
    df = df[df.text.str.len() > 0].drop_duplicates("text").reset_index(drop=True)
    train, rest = train_test_split(df, test_size=0.30, stratify=df.label, random_state=42)
    val, test = train_test_split(rest, test_size=2 / 3, stratify=rest.label, random_state=42)
    for split, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(f"{OUT}/k8_{split}.csv", index=False)


def amazon():
    df = pd.read_csv(io.BytesIO(fetch(AMAZON))).dropna(subset=["reviewText", "overall"])
    star = df["overall"].astype(int)
    label = star.map(lambda s: "negative" if s <= 2 else ("neutral" if s == 3 else "positive"))
    out = pd.DataFrame({"text": df["reviewText"].astype(str).str.strip(), "label": label, "stars": star})
    out[out.text.str.len() > 0].to_csv(f"{OUT}/amazon_cards_test.csv", index=False)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    tweeteval(); k8(); amazon()
    for f in sorted(os.listdir(OUT)):
        d = pd.read_csv(f"{OUT}/{f}")
        print(f"{f:28s} {len(d):6d}  {d.label.value_counts().to_dict()}")
