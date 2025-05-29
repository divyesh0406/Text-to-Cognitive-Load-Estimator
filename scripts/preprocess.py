from datasets import load_dataset
import pandas as pd
import os

# Load dataset
ds = load_dataset("iastate/onestop_english")
label_names = ds['train'].features['label'].names  # ['ele', 'int', 'adv']

# Mapping to cognitive load levels
level_map = {
    "ele": "Low",
    "int": "Medium",
    "adv": "High"
}

data = []
for item in ds['train']:
    text = item['text']
    label_id = item['label']
    label_str = label_names[label_id]  # Convert ID to string label

    final_label = level_map.get(label_str)
    if final_label:
        data.append((text.strip(), final_label))

# Save CSV
df = pd.DataFrame(data, columns=['text', 'label'])
os.makedirs("data/preprocessed", exist_ok=True)
df.to_csv("data/preprocessed/labeled.csv", index=False)

print(f"✅ Saved {len(df)} rows to data/preprocessed/labeled.csv")
