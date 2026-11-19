"""Lexicon-augmented supervised sentiment classifier.

Architecture (scikit-learn Pipeline):

    raw review
      -> Normalizer (slang / elongation / emoji, see normalize.py)
      -> FeatureUnion
           * word TF-IDF, unigrams + bigrams (sublinear tf)      captures "not good", "worth money"
           * character TF-IDF, 2-5 char n-grams within words      robust to typos it has never seen
           * VADER lexicon features [neg, neu, pos, compound]    prior knowledge from a validated lexicon
      -> multinomial Logistic Regression (class_weight="balanced")
      -> P(negative), P(neutral), P(positive)

Why this design: VADER alone is a fixed rule system and cannot learn domain
vocabulary ("heating", "hang", "bakwas"). A linear model over TF-IDF learns it
from labelled data, and the four VADER scores give it a strong prior where
training data is thin. Every component is interpretable: each class has a
weight per word/char-gram, so predictions can be explained.
"""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.nlp.normalize import Normalizer

LABELS = ["negative", "neutral", "positive"]


class NormalizeStep(BaseEstimator, TransformerMixin):
    def __init__(self, slang=None, elongation=False, emoji=False):
        self.slang = slang
        self.elongation = elongation
        self.emoji = emoji

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        norm = Normalizer(self.slang, self.elongation, self.emoji)
        return [norm(t) for t in X]


class VaderFeatures(BaseEstimator, TransformerMixin):
    """[neg, neu, pos, compound] from VADER, as a sparse block."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sia = vader()
        rows = [[s["neg"], s["neu"], s["pos"], s["compound"]] for s in map(sia.polarity_scores, X)]
        return sparse.csr_matrix(np.asarray(rows))


_SIA = None


def vader():
    global _SIA
    if _SIA is None:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        _SIA = SentimentIntensityAnalyzer()
    return _SIA


def vader_label(text, threshold=0.05):
    """Standard VADER decision rule (Hutto & Gilbert, 2014)."""
    c = vader().polarity_scores(text)["compound"]
    return "positive" if c >= threshold else ("negative" if c <= -threshold else "neutral")


def build_features(use_vader=True):
    parts = [
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000,
                                 sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                 max_features=300_000, sublinear_tf=True, lowercase=True)),
    ]
    if use_vader:
        parts.append(("vader", VaderFeatures()))
    return FeatureUnion(parts)


def build_classifier(C=1.0):
    return LogisticRegression(C=C, max_iter=3000, class_weight="balanced")


def build_pipeline(slang="v2", elongation=True, emoji=True, use_vader=True, C=1.0):
    return Pipeline([
        ("normalize", NormalizeStep(slang, elongation, emoji)),
        ("features", build_features(use_vader)),
        ("clf", build_classifier(C)),
    ])
