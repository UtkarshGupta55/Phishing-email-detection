from flask import Flask, render_template, request
from joblib import load
import numpy as np
import re
import spacy
model = load('phishing_model_spacy.pkl')
vectorizer = load('vectorizer_spacy.pkl')
nlp = spacy.load("en_core_web_sm")
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop]
    return ' '.join(tokens)
def svo_structure_spacy(text):
    if not isinstance(text, str) or len(text) == 0:
        return 1
    doc = nlp(text)
    subjects = any(token.dep_ in ["nsubj", "nsubjpass"] for token in doc)
    objects = any(token.dep_ in ["dobj", "pobj"] for token in doc)
    verbs = any(token.pos_ == "VERB" for token in doc)
    return 0 if (subjects and objects and verbs) else 1
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        email_text = request.form['email_text']
        processed_email = preprocess_text(email_text)
        svo_feature = np.array([svo_structure_spacy(processed_email)])
        email_vec = vectorizer.transform([processed_email]).toarray()
        email_features = np.column_stack((email_vec, svo_feature))
        prediction = model.predict(email_features)[0]
        if prediction == 1:
            result = "Phishing Email"
        else:
            result = "Safe Email"
    return render_template('index.html', result=result)
if __name__ == "__main__":
    app.run(debug=True)
