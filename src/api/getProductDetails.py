import requests

def get_product_details(category, product_id=None):
    url = "https://4hcyuavh84.execute-api.ap-northeast-1.amazonaws.com/dev/productdetails"
  
    params = {
        "category": category
    }
    if product_id is not None:
        params["product_id"] = product_id

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for non-2xx status codes

        data = response.json()
        return data

    except requests.exceptions.RequestException as err:
        print("API request failed:", err)

# Example usage
category = "electronics"
product_id = "1234"
results = get_product_details(category, product_id)
print(results)
