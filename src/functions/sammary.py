from transformers import pipeline, BartTokenizer
import openai
import json

def generate_summary(reviews):
    summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = []
    for review in reviews:
        summary = summarization_pipeline(review['input'], max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return summaries

# OpenAI API configuration
openai.api_key = 'YOUR_API_KEY'  # Replace with your OpenAI API key

def ask_chat_gpt(question, context):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=f"Question: {question}\nContext: {context}\nAnswer:",
        temperature=0.7,
        max_tokens=50,
        n=1,
        stop=None,
    )
    return response.choices[0].text.strip()

def review_sammary(reviews):
    
    product_reviews = [record['review'] for record in reviews]
    summaries = generate_summary(product_reviews)

    opinions = []
    for i, record in enumerate(data):
        question = "based on given review summary tell me Should I buy this product?"  # Customize the question if needed
        response = ask_chat_gpt(question, summaries[i])
        opinions.append({"review": record['review'], "sentiment": record['sentiment'], "opinion": response})

    return response

# Example usage

# for opinion in opinions:
#     print("Review:", opinion['input'])
#     print("Sentiment:", opinion['sentiment'])
#     print("Opinion:", opinion['opinion'])
#     print()
