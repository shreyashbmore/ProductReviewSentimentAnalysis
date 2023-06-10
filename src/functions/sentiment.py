import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from src.functions.sarcasm import sarcasm_correction
from src.functions.preprocess import preprocess_text
sia = SentimentIntensityAnalyzer()


def detect_sentiment(sentence):
    sentence = preprocess_text(sentence)
    # Analyze the sentiment of the sentence using VADER
    sentiment_scores = sia.polarity_scores(sentence)

    # Determine the sentiment label based on the compound score
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        sentiment_label = 'positive'
    elif compound_score <= -0.05:
        sentiment_label = 'negative'
    else:
        sentiment_label = 'neutral'
    isCorrection = sarcasm_correction(sentence)
    if isCorrection != "no":
        sentiment_label = isCorrection
    return sentiment_label