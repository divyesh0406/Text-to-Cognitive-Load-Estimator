import streamlit as st
from transformers.models import BertTokenizer, BertForSequenceClassification
import torch

st.title("Text-to-Cognitive Load Estimator")

model = BertForSequenceClassification.from_pretrained("models/cognitive_bert")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = st.text_area("Enter your text")

if st.button("Estimate Load"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
    st.write(f"**Cognitive Load**: {['Low', 'Medium', 'High'][pred]}")
    st.balloons()
    st.write("Thank you for using the Cognitive Load Estimator!")