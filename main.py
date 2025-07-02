# main.py

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import sigmoid

# Load fine-tuned model and tokenizer
model_path = "bert_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Required predict() function
def predict(csv_path):
    df = pd.read_csv(csv_path)
    texts = df["text"].tolist()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = sigmoid(logits)
        predictions = (probs >= 0.5).int().tolist()

    return predictions
