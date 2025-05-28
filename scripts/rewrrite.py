from transformers import pipeline

simplifier = pipeline("text2text-generation", model="t5-small")

def simplify(text):
    prompt = f"Simplify: {text}"
    return simplifier(prompt, max_length=100)[0]['generated_text']
