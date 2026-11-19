"""Benchmark: does slang normalization help, and does the v2 model beat v1?

Design
------
Train:  TweetEval train (3-class tweets) + K8 train (binary product reviews)
Tune:   C of Logistic Regression on TweetEval val + K8 val (test sets untouched)
Test:   TweetEval test (3-class), K8 test (binary), Amazon memory-card reviews
        (3-class, never seen in training at all: out-of-domain check)

Factors
-------
Normalization:  none | slang-v1 (original list) | slang-v2 (v1 list + e-commerce/Hinglish
                extension) | slang-v2 + elongation + emoji (full)
Classifier:     VADER rules | TF-IDF + LogReg | TF-IDF + VADER features + LogReg (hybrid)
Plus the exact v1 pipeline as a reference row.

Statistics
----------
Macro-F1 on each test set and on its "slang subset" (reviews containing at least one
lexicon term). Differences are tested with a paired bootstrap (2,000 resamples):
95% CI of the macro-F1 difference and a one-sided p-value.

Run:  python scripts/download_data.py && python scripts/evaluate.py
Output: reports/results.json, reports/RESULTS.md, reports/confusion_*.png,
        models/sentiment_model.joblib (the production model).
"""
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.legacy.pipeline_v1 import compound_v1  # noqa: E402
from src.nlp.model import (LABELS, NormalizeStep, build_classifier,  # noqa: E402
                           build_features, vader)
from src.nlp.normalize import Normalizer  # noqa: E402

P = "data/processed"
RNG = np.random.default_rng(0)
N_BOOT = 2000
C_GRID = [0.5, 1.0, 2.0]

NORMS = {
    "none": dict(slang=None, elongation=False, emoji=False),
    "slang-v1": dict(slang="v1", elongation=False, emoji=False),
    "slang-v2": dict(slang="v2", elongation=False, emoji=False),
    "slang-v2+elong+emoji": dict(slang="v2", elongation=True, emoji=True),
}


def load(name):
    return pd.read_csv(f"{P}/{name}.csv").dropna(subset=["text"]).reset_index(drop=True)


DATA = {
    "train": pd.concat([load("tweeteval_train"), load("k8_train")], ignore_index=True),
    "val_tweet": load("tweeteval_val"), "val_k8": load("k8_val"),
    "tweeteval": load("tweeteval_test"), "k8": load("k8_test"), "amazon": load("amazon_cards_test"),
}
TESTS = {"tweeteval": "3-class", "k8": "binary", "amazon": "3-class"}


# ---------- decision rules -------------------------------------------------
def from_compound(compound, binary):
    if binary:
        return np.where(compound >= 0, "positive", "negative")
    return np.where(compound >= 0.05, "positive", np.where(compound <= -0.05, "negative", "neutral"))


def from_proba(proba, classes, binary):
    classes = np.asarray(classes)
    if binary:  # K8 has no neutral class: choose between negative and positive
        idx = [list(classes).index("negative"), list(classes).index("positive")]
        return np.where(proba[:, idx[1]] >= proba[:, idx[0]], "positive", "negative")
    return classes[proba.argmax(1)]


def macro_f1(y, p, binary):
    labels = ["negative", "positive"] if binary else LABELS
    return f1_score(y, p, labels=labels, average="macro", zero_division=0)


def _fast_macro_f1(y, p, k):
    """Macro-F1 on integer-coded labels (same definition as sklearn, vectorized for the bootstrap)."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return f1.mean()


def metrics(y, p, binary):
    labels = ["negative", "positive"] if binary else LABELS
    out = {"macro_f1": macro_f1(y, p, binary), "accuracy": accuracy_score(y, p), "n": int(len(y))}
    if not binary:
        out["macro_recall"] = recall_score(y, p, labels=labels, average="macro", zero_division=0)
        out["per_class_f1"] = dict(zip(labels, f1_score(y, p, labels=labels, average=None, zero_division=0)))
    return out


def paired_bootstrap(y, a, b, binary):
    """Delta = macroF1(b) - macroF1(a). Returns observed delta, 95% CI, P(delta <= 0)."""
    labels = ["negative", "positive"] if binary else LABELS
    code = {l: i for i, l in enumerate(labels)}
    y, a, b = (np.array([code[v] for v in arr]) for arr in (y, a, b))
    k, n = len(labels), len(y)
    obs = _fast_macro_f1(y, b, k) - _fast_macro_f1(y, a, k)
    deltas = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        deltas[i] = _fast_macro_f1(y[idx], b[idx], k) - _fast_macro_f1(y[idx], a[idx], k)
    return {"delta": obs, "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "p_value": float((deltas <= 0).mean())}


# ---------- run ------------------------------------------------------------
def main():
    t0 = time.time()
    norm_text = {}
    for nname, cfg in NORMS.items():
        step = NormalizeStep(**cfg)
        norm_text[nname] = {k: step.transform(df.text) for k, df in DATA.items()}
        print(f"normalized [{nname}] {time.time() - t0:.0f}s", flush=True)

    preds = {}   # (classifier, norm) -> {test: predictions}
    chosen_c, val_scores, fitted = {}, {}, {}
    sia = vader()

    # 1) exact v1 pipeline
    preds[("v1-pipeline", "v1-preprocess")] = {
        t: from_compound(np.array([compound_v1(x) for x in DATA[t].text]), TESTS[t] == "binary") for t in TESTS}

    for nname in NORMS:
        T = norm_text[nname]
        # 2) VADER rules on normalized text
        preds[("vader", nname)] = {
            t: from_compound(np.array([sia.polarity_scores(x)["compound"] for x in T[t]]), TESTS[t] == "binary")
            for t in TESTS}
        # 3) supervised models
        for cname, use_vader in [("tfidf-logreg", False), ("hybrid", True)]:
            feats = build_features(use_vader)
            Xtr = feats.fit_transform(T["train"])
            Xv = {k: feats.transform(T[k]) for k in ["val_tweet", "val_k8", *TESTS]}
            best = None
            for C in C_GRID:
                clf = build_classifier(C).fit(Xtr, DATA["train"].label)
                score = np.mean([
                    macro_f1(DATA["val_tweet"].label, from_proba(clf.predict_proba(Xv["val_tweet"]), clf.classes_, False), False),
                    macro_f1(DATA["val_k8"].label, from_proba(clf.predict_proba(Xv["val_k8"]), clf.classes_, True), True)])
                if best is None or score > best[0]:
                    best = (score, C, clf)
            _, C, clf = best
            chosen_c[f"{cname}|{nname}"] = C
            val_scores[f"{cname}|{nname}"] = best[0]
            fitted[(cname, nname)] = (feats, clf)
            preds[(cname, nname)] = {t: from_proba(clf.predict_proba(Xv[t]), clf.classes_, TESTS[t] == "binary")
                                     for t in TESTS}
            print(f"trained [{cname} | {nname}] C={C} val={best[0]:.4f} {time.time() - t0:.0f}s", flush=True)

    # production model = best supervised config on VALIDATION (test sets play no part in the choice)
    production = tuple(max(val_scores, key=val_scores.get).split("|"))
    print(f"production model (best on validation): {production}", flush=True)
    feats, clf = fitted[production]
    for _, vec in feats.transformer_list:
        if hasattr(vec, "stop_words_"):
            vec.stop_words_ = None  # not needed for inference; keeps the file small
    model = Pipeline([("normalize", NormalizeStep(**NORMS[production[1]])), ("features", feats), ("clf", clf)])
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/sentiment_model.joblib", compress=3)
    del fitted

    # ---------- scoring ----------
    subset_norm = Normalizer("v2")
    subsets = {t: np.array([len(subset_norm.slang_hits(x)) > 0 for x in DATA[t].text]) for t in TESTS}
    results = {"configs": {}, "significance": {}, "chosen_C": chosen_c, "val_scores": val_scores,
               "production": "|".join(production),
               "subset_sizes": {t: int(m.sum()) for t, m in subsets.items()}}
    for (cname, nname), pt in preds.items():
        key = f"{cname}|{nname}"
        results["configs"][key] = {}
        for t in TESTS:
            y, p, b, m = DATA[t].label.values, pt[t], TESTS[t] == "binary", subsets[t]
            results["configs"][key][t] = metrics(y, p, b)
            results["configs"][key][t + "_slang_subset"] = metrics(y[m], p[m], b)

    comparisons = [
        ("v1-pipeline|v1-preprocess", "|".join(production), "v1 system -> v2 system"),
        ("vader|none", "v1-pipeline|v1-preprocess", "VADER: v1 preprocessing (as shipped)"),
        ("vader|none", "vader|slang-v1", "VADER: v1 slang list"),
        ("vader|none", "vader|slang-v2", "VADER: extended slang lexicon"),
        ("vader|slang-v1", "vader|slang-v2", "VADER: extension over v1 list"),
        ("hybrid|none", "hybrid|slang-v1", "Hybrid: v1 slang list"),
        ("hybrid|none", "hybrid|slang-v2", "Hybrid: extended slang lexicon"),
        ("hybrid|none", "hybrid|slang-v2+elong+emoji", "Hybrid: full normalization"),
        ("tfidf-logreg|none", "tfidf-logreg|slang-v2", "TF-IDF LogReg: extended slang lexicon"),
        ("hybrid|slang-v2", "hybrid|slang-v2+elong+emoji", "Hybrid: adding elongation + emoji"),
        ("tfidf-logreg|slang-v2", "hybrid|slang-v2", "Adding VADER features"),
    ]
    for a, b, label in comparisons:
        ka, kb = tuple(a.split("|")), tuple(b.split("|"))
        for t in TESTS:
            y, binary, m = DATA[t].label.values, TESTS[t] == "binary", subsets[t]
            results["significance"][f"{label} :: {t}"] = paired_bootstrap(y, preds[ka][t], preds[kb][t], binary)
            results["significance"][f"{label} :: {t} (slang subset)"] = paired_bootstrap(
                y[m], preds[ka][t][m], preds[kb][t][m], binary)

    # qualitative examples: slang-subset reviews fixed by normalization (hybrid none -> hybrid full)
    examples = []
    for t in ["k8", "tweeteval"]:
        y, m = DATA[t].label.values, subsets[t]
        a, b = preds[("hybrid", "none")][t], preds[production][t]
        for i in np.where(m & (a != y) & (b == y))[0][:8]:
            examples.append({"set": t, "text": DATA[t].text[i], "gold": y[i], "before": a[i], "after": b[i],
                             "normalized": norm_text[production[1]][t][i]})
    results["fixed_examples"] = examples

    os.makedirs("reports", exist_ok=True)
    with open("reports/results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    write_markdown(results)
    confusion_plots(preds[production], "|".join(production))
    print(f"done in {time.time() - t0:.0f}s")


def write_markdown(r):
    L = ["# Evaluation results", "",
         "Generated by `scripts/evaluate.py`. Macro-F1 unless noted. "
         "TweetEval also reports macro-recall (its official metric). "
         "K8 is binary (positive vs negative). Amazon memory-card reviews are fully out-of-domain.", "",
         f"Production model (selected by validation score): **{r['production']}**", "",
         f"Slang-subset sizes (test reviews containing at least one lexicon term): {r['subset_sizes']}", "",
         "## Main table", "",
         "| Classifier | Normalization | TweetEval F1 | TweetEval rec. | K8 F1 | Amazon F1 | TweetEval F1 (slang) | K8 F1 (slang) | Amazon F1 (slang) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for key, v in r["configs"].items():
        c, n = key.split("|")
        L.append(f"| {c} | {n} | {v['tweeteval']['macro_f1']:.3f} | {v['tweeteval']['macro_recall']:.3f} | "
                 f"{v['k8']['macro_f1']:.3f} | {v['amazon']['macro_f1']:.3f} | "
                 f"{v['tweeteval_slang_subset']['macro_f1']:.3f} | {v['k8_slang_subset']['macro_f1']:.3f} | "
                 f"{v['amazon_slang_subset']['macro_f1']:.3f} |")
    L += ["", "## Significance (paired bootstrap, 2,000 resamples)", "",
          "| Comparison | Test set | Δ macro-F1 | 95% CI | p (Δ ≤ 0) |", "|---|---|---|---|---|"]
    for key, s in r["significance"].items():
        label, t = key.split(" :: ")
        L.append(f"| {label} | {t} | {s['delta']:+.3f} | [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}] | {s['p_value']:.3f} |")
    L += ["", "## Reviews fixed by normalization (hybrid model, before → after)", ""]
    for e in r["fixed_examples"]:
        L.append(f"- [{e['set']}] \"{e['text'][:160]}\" — gold **{e['gold']}**, {e['before']} → {e['after']}")
    L += ["", "## Validation scores used for model selection", "",
          "| Config | mean val macro-F1 | C |", "|---|---|---|"]
    for k, v in r["val_scores"].items():
        L.append(f"| {k} | {v:.4f} | {r['chosen_C'][k]} |")
    with open("reports/RESULTS.md", "w") as f:
        f.write("\n".join(L) + "\n")


def confusion_plots(pred, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, t in zip(axes, TESTS):
        labels = ["negative", "positive"] if TESTS[t] == "binary" else LABELS
        ConfusionMatrixDisplay.from_predictions(DATA[t].label, pred[t], labels=labels, ax=ax,
                                                colorbar=False, cmap="Blues")
        ax.set_title({"tweeteval": "TweetEval test", "k8": "K8 reviews test", "amazon": "Amazon (out-of-domain)"}[t])
    fig.suptitle(f"Production model: {name}")
    fig.tight_layout()
    fig.savefig("reports/confusion_matrices.png", dpi=130)


if __name__ == "__main__":
    main()
