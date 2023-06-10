from flask import Flask, render_template, request, session
import json
from src.functions.sentiment import detect_sentiment
from src.api.postReview import execute_api
from src.api.getReview import execute_get_api
from src.functions.summary import review_summary
app = Flask(__name__, static_url_path='/static')  
app.secret_key = '12345'


@app.route('/')
def home():
    return render_template('index2.html', show_results=False)


@app.route("/category")
def category():
    return render_template('category.html')


@app.route("/electronics_products", methods=['GET', 'POST'])
def electronics():
    session["id"]="12345"
    if request.method == 'POST':
        user_input = request.form['user_input']
        sentiment_label = detect_sentiment(user_input)
        print("this is analyzed sentiment"+sentiment_label)
        body = {
         "product_id": "12345",
         "review": user_input,
        "sentiment":sentiment_label 
     }
        response = execute_api(body)
        
    return render_template('electronics_products.html')

@app.route("/clothing_products", methods=['GET', 'POST'])
def clothing():
    session["id"]="1234"
    if request.method == 'POST':
        user_input = request.form['user_input']
        sentiment_label = detect_sentiment(user_input)
        print("this is analyzed sentiment"+sentiment_label)
        body = {
         "product_id": "1234",
         "review": user_input,
        "sentiment":sentiment_label 
     }
        response = execute_api(body)

    return render_template('clothing_products.html')

@app.route("/furniture_products", methods=['GET', 'POST'])
def furniture():
    session["id"]="123"
    if request.method == 'POST':
        user_input = request.form['user_input']
        sentiment_label = detect_sentiment(user_input)
        print("this is analyzed sentiment"+sentiment_label)
        body = {
         "product_id": "123",
         "review": user_input,
        "sentiment":sentiment_label 
     }
        response = execute_api(body)

    return render_template('furniture_products.html')


@app.route("/review_analysis", methods=['GET'])
def review_analysisf():
    id = session["id"]
    reviews = execute_get_api(id)
    #reviews = [{'sentiment': 'positive', 'review': 'this is best product'}, {'sentiment': 'positive', 'review': 'this is nice product'}, {'sentiment': 'negative', 'review': 'camera quality is not upto the mark'}, {'sentiment': 'neutral', 'review': 'display is good but battery is bad'}]
    review_type = request.args.get('reviewType')
    sentiment_count = {'positive': 0, 'negative': 0, 'neutral': 0}

    for item in reviews:
        sentiment = item['sentiment']
        sentiment_count[sentiment] += 1

    
    filtered_reviews = []
    if review_type:
        filtered_reviews = [review for review in reviews if review['sentiment'] == review_type]
    else:
        filtered_reviews = reviews
    reviews = filtered_reviews
    return render_template('reviewpage.html', reviews=reviews, sentiment_count=sentiment_count)
        

@app.route("/opinion")
def opinion():
    id = session["id"]
    reviews = execute_get_api(id)
    # reviews = [{'sentiment': 'positive', 'review': 'this is best product'}, {'sentiment': 'positive', 'review': 'this is nice product'}, {'sentiment': 'negative', 'review': 'camera quality is not upto the mark'}, {'sentiment': 'neutral', 'review': 'display is good but battery is bad'}]
    opinion = review_summary(reviews)
    
    return render_template('opinion.html ', opinion=opinion)


if __name__ == '__main__':
    app.run(debug=True)