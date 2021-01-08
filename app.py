from flask import Flask, render_template, request
import jsonify
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from joblib import load
import sklearn
import warnings
import re
app = Flask(__name__)
warnings.filterwarnings("ignore")

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


cv_2 = load("cv_2.joblib")
loaded_model = load('model_updated.joblib') 



from sklearn.feature_extraction.text import TfidfVectorizer
def new_review(new_review):
    new_review = new_review
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    all_stopwords = new_review
    new_review = all_stopwords
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    new_X_test = cv_2.transform(new_corpus).toarray()
    pred = loaded_model.predict(new_X_test)
    return pred





@app.route("/classify", methods=['POST'])
def classify():

    review_text=""

    if request.method == 'POST':

        review_text=request.form['enter_review']
        final_review = new_review(review_text)

        if final_review == 1:
            output="Positive"
        elif final_review==0:
            output="Neutral"
        elif final_review==-1:
            output="Negative"

        
        return render_template('index.html',classification_text="Your Review is {}".format(output))

    else:

        return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)


