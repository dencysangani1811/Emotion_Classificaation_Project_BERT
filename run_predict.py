# run_predict.py

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import sigmoid

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert_model")
tokenizer = BertTokenizer.from_pretrained("bert_model")
model.eval()

# Read your input CSV
df = pd.read_csv("sample.csv")  # Make sure this file exists!
texts = df["text"].tolist()

# Tokenize
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    probs = sigmoid(outputs.logits).numpy()
    preds = (probs >= 0.5).astype(int)

# Convert to DataFrame
emotion_labels = ["anger", "fear", "joy", "sadness", "surprise"]
df_preds = pd.DataFrame(preds, columns=emotion_labels)

# Combine and print
df_combined = pd.concat([df, df_preds], axis=1)
print(df_combined)