import pandas as pd
import re
from textstat import flesch_kincaid_grade

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def label_text(text):
    score = flesch_kincaid_grade(text)
    if score < 5:
        return "Low"
    elif score < 10:
        return "Medium"
    else:
        return "High"

def preprocess(df):
    df['text'] = df['text'].apply(clean_text)
    df['label'] = df['text'].apply(label_text)
    return df
