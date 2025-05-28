import torch
import pandas as pd

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset


def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Load your custom dataset
df = pd.read_csv("data/processed/labeled.csv")
dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.2)

training_args = TrainingArguments(
    output_dir="./models",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

trainer.train()
model.save_pretrained("models/cognitive_bert")
