import requests

def execute_api(body):
    url = "https://oswkj1ydx8.execute-api.ap-northeast-1.amazonaws.com/dev/review"  # Replace with your API endpoint URL

    # Request body
    # body = {
    #     "product_id": "123",
    #     "review": "it is having great display",
    #     "sentiment": "positive"
    # }

    try:
        response = requests.post(url, json=body)
        response.raise_for_status()  # Raise an exception for non-2xx status codes

        # Process the response data if needed
        data = response.json()
        print("API response:", data)

    except requests.exceptions.RequestException as err:
        print("API request failed:", err)

# Call the function to execute the API
# execute_api()
