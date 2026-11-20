import os

from flask import Flask, render_template, request, session

from src.api.storage import get_reviews, save_review
from src.functions.sentiment import analyze
from src.functions.summary import distribution, review_summary

app = Flask(__name__, static_url_path='/static')
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")

PRODUCTS = {"electronics": "12345", "clothing": "1234", "furniture": "123"}


@app.route('/')
def home():
    return render_template('index2.html', show_results=False)


@app.route("/category")
def category():
    return render_template('category.html')


def product_page(category_name):
    product_id = PRODUCTS[category_name]
    session["id"] = product_id
    result = None
    if request.method == 'POST':
        user_input = request.form.get('user_input', '').strip()
        if user_input:
            result = analyze(user_input)
            save_review({"product_id": product_id, "review": user_input, "sentiment": result["label"],
                         "confidence": result["confidence"]})
    return render_template(f'{category_name}_products.html', result=result)


@app.route("/electronics_products", methods=['GET', 'POST'])
def electronics():
    return product_page("electronics")


@app.route("/clothing_products", methods=['GET', 'POST'])
def clothing():
    return product_page("clothing")


@app.route("/furniture_products", methods=['GET', 'POST'])
def furniture():
    return product_page("furniture")


@app.route("/review_analysis", methods=['GET'])
def review_analysis():
    reviews = get_reviews(session.get("id", PRODUCTS["electronics"]))
    sentiment_count, sentiment_pct, _ = distribution(reviews)
    review_type = request.args.get('reviewType')
    if review_type:
        reviews = [r for r in reviews if r.get('sentiment') == review_type]
    return render_template('reviewpage.html', reviews=reviews,
                           sentiment_count=sentiment_count, sentiment_pct=sentiment_pct)


@app.route("/opinion")
def opinion():
    reviews = get_reviews(session.get("id", PRODUCTS["electronics"]))
    return render_template('opinion.html', opinion=review_summary(reviews))


if __name__ == '__main__':
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
