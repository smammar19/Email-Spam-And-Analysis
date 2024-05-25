from flask import Flask, request, render_template, send_from_directory, url_for
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import lime
import lime.lime_text
import sklearn
from sklearn.pipeline import make_pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

app = Flask(__name__, static_folder='static')
df = pd.read_csv('spam.csv')

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Initialize stemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

def explain_prediction(text, vectorizer, model):
    pipeline = make_pipeline(vectorizer, model)
    explainer = lime.lime_text.LimeTextExplainer(class_names=['Not Spam', 'Spam'])
    explanation = explainer.explain_instance(text, pipeline.predict_proba, num_features=6)
    print("Explanation from explain_prediction:", explanation) 
    return explanation




def generate_bar_chart(explanation):
    words = [word for word, _ in explanation.as_list(label=1)]
    weights = [weight for _, weight in explanation.as_list(label=1)]
    plt.figure(figsize=(8, 6))
    plt.bar(words, weights)
    plt.xticks(rotation=45)
    plt.xlabel('Words')
    plt.ylabel('Weight')
    plt.title('Word Importance')
    plt.tight_layout()
    # Save the chart as an image file
    plt.savefig('static/word_importance.png', bbox_inches='tight')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/static/freq.png')
def serve_static():
    return send_from_directory(app.static_folder, 'freq.png')

@app.route('/static/confusion.png')
def serve_confusion_matrix():
    return send_from_directory(app.static_folder, 'confusion.png')

@app.route('/static/cnf_svc.png')
def serve_confusion_matrix_stacking():
    return send_from_directory(app.static_folder, 'cnf_svc.png')

@app.route('/dataset_pie_chart')
def dataset_pie_chart():
    return send_from_directory('static', 'dataset pie chart.png')

@app.route('/static/cnf_mnb.png')
def serve_confusion_matrix_mnb():
    return send_from_directory(app.static_folder, 'cnf_mnb.png')

@app.route('/static/cnf_etc.png')
def serve_confusion_matrix_etc():
    return send_from_directory(app.static_folder, 'cnf_etc.png')


@app.route('/predict', methods=['POST'])
def predict():
    input_sms = request.form['sms_text']
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    result_proba = model.predict_proba(vector_input)[0]
    spam_probability = result_proba[1]
    not_spam_probability = result_proba[0]
    if result == 1:
        prediction = "Spam"
    else:
        prediction = "Not Spam"
    text_length = len(input_sms)
    explanation = explain_prediction(input_sms, tfidf, model)
    explanation_list = explanation.as_list(label=1)
    if explanation_list:
                    explanation_text = [{'word': word, 'weight': weight, 'label': 'Spam'} for word, weight in explanation_list]   
    else:
                    explanation_text = []
    print("Explanation Text:", explanation_text)

    generate_bar_chart(explanation)
    word_importance_chart = 'static/word_importance.png'
    return render_template('index.html', prediction=prediction, text_length=text_length,
                           spam_probability=spam_probability, not_spam_probability=not_spam_probability,
                           explanation_text=explanation_text, word_importance_chart=word_importance_chart)


if __name__ == '__main__':
    app.run(debug=True)


