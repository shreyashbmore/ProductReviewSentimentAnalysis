import os

import requests

def execute_get_api(product_id):
    url = os.environ.get("REVIEW_API_URL", "https://oswkj1ydx8.execute-api.ap-northeast-1.amazonaws.com/dev/review")

    # Query parameters
    params = {
        "product_id": product_id
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise an exception for non-2xx status codes

        # Process the response data if needed
        data = response.json()
        print(data)
        return data

    except requests.exceptions.RequestException as err:
        print("API request failed:", err)
        return []

# Call the function to execute the GET API
# product_id = "1234"  # Replace with the desired product ID
# execute_get_api(product_id)
