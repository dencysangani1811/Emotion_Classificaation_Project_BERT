
# train_bert.py

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score, hamming_loss
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments
from transformers import DataCollatorWithPadding, Trainer
from datasets import Dataset

# Load and prepare data
df = pd.read_csv("C:/Users/Nilesh/Desktop/DS/Sem 3/NLP/Emotion Project/track-a.csv")
df["labels"] = df[["anger", "fear", "joy", "sadness", "surprise"]].values.tolist()
df = df[["text", "labels"]]

# Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokenize(example):
    return tokenizer(example["text"], truncation=True)
dataset = dataset.map(tokenize, batched=True)

# Train-Test Split
dataset = dataset.train_test_split(test_size=0.2)

# Convert labels to float32 arrays (not just float lists!)
def format_labels(example):
    example["labels"] = np.array(example["labels"], dtype=np.float32)
    return example

train_dataset = dataset["train"].map(format_labels)
eval_dataset = dataset["test"].map(format_labels)

# Model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5,
    problem_type="multi_label_classification"
)

# Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics
def compute_metrics(p):
    preds = torch.sigmoid(torch.tensor(p.predictions)).numpy()
    preds = (preds >= 0.5).astype(int)
    labels = p.label_ids
    return {
        "f1": f1_score(labels, preds, average="macro"),
        "hamming_loss": hamming_loss(labels, preds)
    }

# Training Arguments
training_args = TrainingArguments(
    output_dir="./bert_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_strategy="epoch",
    save_total_limit=1
)

# ✅ Custom Trainer to fix float issue
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(torch.float32)  # <== critical fix
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Trainer Instance
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 🚀 Train and Save
trainer.train()
trainer.save_model("./bert_model")
tokenizer.save_pretrained("./bert_model")
