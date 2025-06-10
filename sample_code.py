

# legal_ai_app.py

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
# Load Legal-BERT model and tokenizer
model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_legal_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# Streamlit UI
st.title("Legal AI Judgment Assistant")
st.write("Enter a legal document or case summary below:")

user_input = st.text_area("Legal Text", height=200)

if st.button("Analyze"):
    if user_input.strip():
        prediction = analyze_legal_text(user_input)
        st.write(f"Predicted legal class (example): {prediction}")
    else:
        st.warning("Please enter some legal text.")
