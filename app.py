
from flask import Flask, render_template, request, jsonify
import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

nltk.download('punkt')

app = Flask(__name__)

# Load FAQs
with open('data/faqs.json', 'r') as f:
    faq_data = json.load(f)

questions = [faq['question'] for faq in faq_data]
answers = [faq['answer'] for faq in faq_data]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get', methods=['POST'])
def chatbot_response():
    user_question = request.form['msg']
    user_input_clean = clean_text(user_question)
    user_vec = vectorizer.transform([user_input_clean])

    similarity = cosine_similarity(user_vec, question_vectors)
    index = similarity.argmax()
    confidence = similarity[0][index]

    if confidence > 0.3:
        return jsonify({"reply": answers[index]})
    else:
        return jsonify({"reply": "Sorry, I couldn't understand that. Please rephrase."})


if __name__ == '__main__':
    app.run(debug=True)
