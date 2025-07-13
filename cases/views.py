import os
import re
import joblib
from django.shortcuts import render, redirect
from django.conf import settings

TRANSFORMERS_AVAILABLE = True
LANGTRANS_AVAILABLE = True

try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForSequenceClassification
    import torch
except ModuleNotFoundError:
    TRANSFORMERS_AVAILABLE = False

try:
    from langdetect import detect
    from googletrans import Translator
except ModuleNotFoundError:
    LANGTRANS_AVAILABLE = False

if TRANSFORMERS_AVAILABLE:
    logistic_model_path = os.path.join(settings.BASE_DIR, 'models/court_case_model.pkl')
    tfidf_vectorizer_path = os.path.join(settings.BASE_DIR, 'models/tfidf_vectorizer.pkl')
    t5_model_path = os.path.join(settings.BASE_DIR, 'models/t5_model/')

    logistic_model = joblib.load(logistic_model_path)
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)

    legalbert_tokenizer = BertTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
    legalbert_model = BertForSequenceClassification.from_pretrained('nlpaueb/legal-bert-base-uncased')

    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path)

if LANGTRANS_AVAILABLE:
    translator = Translator()

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def ensure_english(text):
    if LANGTRANS_AVAILABLE:
        try:
            lang = detect(text)
            if lang != 'en':
                translated = translator.translate(text, src=lang, dest='en')
                return translated.text
        except Exception:
            pass
    return text

def extract_legal_entities(text):
    if TRANSFORMERS_AVAILABLE:
        tokens = legalbert_tokenizer.tokenize(text)
        return list(set([token for token in tokens if len(token) > 4]))
    return []

def predict_outcome(text):
    if TRANSFORMERS_AVAILABLE:
        features = tfidf_vectorizer.transform([text])
        prediction = logistic_model.predict(features)[0]
        confidence = max(logistic_model.predict_proba(features)[0])
        return prediction, confidence
    return 'unknown', 0.0

def summarize_text(text):
    if TRANSFORMERS_AVAILABLE:
        inputs = t5_tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        outputs = t5_model.generate(inputs["input_ids"], max_length=200, num_beams=4, early_stopping=True)
        return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return "Summary not available due to missing dependencies."

# Views
def index(request):
    return render(request, 'index.html')

def inference(request):
    if not TRANSFORMERS_AVAILABLE:
        return render(request, 'error.html', {
            'error_message': "Required libraries ('transformers', 'torch') are not installed. Please run 'pip install transformers torch'."
        })

    if request.method == 'POST':
        input_text = request.POST.get('input_text', '')
        input_text = ensure_english(input_text)
        cleaned_text = clean_text(input_text)

        prediction, confidence = predict_outcome(cleaned_text)
        summary = summarize_text(input_text)
        legal_entities = extract_legal_entities(input_text)

        if prediction == 'positive':
            message = f"Plaintiff likely to win. Confidence: {confidence:.2f}. Strong evidence in favor."
        elif prediction == 'negative':
            message = f"Defendant likely to win. Confidence: {confidence:.2f}. Legal arguments favor defendant."
        else:
            message = f"Outcome is uncertain. Confidence: {confidence:.2f}. Complex case with mixed factors."

        results = {
            'logistic_prediction': prediction,
            'logistic_description': message,
            't5_summary': summary,
            'legal_entities': ', '.join(legal_entities),
        }

        return render(request, 'result.html', results)

    return redirect('index')
