from transformers import pipeline, BartTokenizer
import openai
import json

# OpenAI API configuration
openai.api_key = 'YOUR_API_KEY'  # Replace with your OpenAI API key

# Initialize the summarization pipeline outside the function


def concatenate_reviews(reviews):
    concatenated_reviews = set()
    result = ''

    for review in reviews:
        sentence = review['review']
        if sentence not in concatenated_reviews:
            result += sentence + '. '  # Add a period after each review
            concatenated_reviews.add(sentence)

    return result.strip()

def ask_chat_gpt(question, context):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=f"Question: {question}\nContext: {context}\nAnswer:",
        temperature=0.7,
        max_tokens=300,
        n=1,
        stop=None,
    )
    return response.choices[0].text.strip()

def review_summary(reviews):
    context = concatenate_reviews(reviews)
   
    question = "Based on the given reviews can I buy this product? and why"  
    response = ask_chat_gpt(question, context)
    return response



