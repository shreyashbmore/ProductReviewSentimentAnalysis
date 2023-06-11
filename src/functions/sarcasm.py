import openai

# Set up your OpenAI API credentials
openai.api_key = 'YOUR_API_KEY'


def sarcasm_correction(sentence):
    prompt = f"give sentiment of sentence\n\"{sentence}\""
    
    # Generate a response from ChatGPT
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=5,  # Set an appropriate value based on response length
        temperature=0.5,  # Adjust temperature to control response randomness
        n=1,  # Generate a single response
        stop=None,  # Don't stop generation prematurely
    )
    
    # Extract the generated response
    generated_response = response.choices[0].text.strip()
    
    sentiment_mapping = {
        "Positive": "positive",
        "Negative": "negative",
        "Neutral": "neutral"
    }
    
    sentiment = sentiment_mapping.get(generated_response, "no")
    # sentiment = generated_response
    return sentiment
