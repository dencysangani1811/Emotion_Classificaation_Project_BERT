# 🤖 BERT-Based Multi-Label Emotion Classification

## 📌 Description

This project implements a **multi-label emotion classification** model using a fine-tuned **BERT** transformer. Each input text can express multiple emotions from the following five categories:

- 😠 Anger
- 😨 Fear
- 😄 Joy
- 😢 Sadness
- 😲 Surprise

The final pipeline loads a `.csv` with text input and returns a 2D array of predictions per sample, e.g., `[0, 1, 0, 1, 0]`.

---

## ⚙️ Implementation Steps

### 1. **Dataset Preparation**
- Source: `track-a.csv`
- Format: `text` + five binary columns for each emotion.
- Combined all five binary columns into a single `labels` column → a list of 0s and 1s per row.

---

### 2. **Text Tokenization**
- Used Hugging Face’s `bert-base-uncased` tokenizer.
- Applied truncation and dynamic padding.
- Used Hugging Face’s `datasets` library to wrap and tokenize efficiently with `.map()`.

---

### 3. **Train-Test Split**
- Split the dataset into 80% training and 20% test data.
- Converted label arrays to float for compatibility with PyTorch's loss function (`BCEWithLogitsLoss`).

---

### 4. **Model Configuration**
- Model: `BertForSequenceClassification` (from `transformers`)
- Settings:
  - `num_labels = 5`
  - `problem_type = "multi_label_classification"`
- This allows independent sigmoid outputs per emotion.

---

### 5. **Training**
- Used Hugging Face’s `Trainer` API with the following settings:
  - `epochs = 4`
  - `batch size = 8`
  - `learning rate = 2e-5`
  - Evaluation and logging every epoch
- Defined a custom `compute_metrics()` function that returns:
  - **Macro F1-Score**
  - **Hamming Loss**

📷 Screenshots of training results are stored in `ss_results/`.

---

### 6. **Prediction Pipeline**
- Defined in `main.py`, the `predict()` function:
  - Takes a CSV path as input
  - Loads and tokenizes the text
  - Loads the trained model
  - Applies sigmoid + threshold (≥ 0.5) to produce predictions
  - Returns a NumPy array of shape `(n_samples, 5)`

- Sample test execution is available via `run_predict.py`.

---

### 7. **Model Size Consideration**
- The trained BERT model was ~1.2GB, exceeding GitHub's 100MB file limit.
- Hence, the `bert_model/` folder with heavy checkpoints was excluded from the repository.
- However:
  - All scripts to retrain and regenerate the model are included.
  - Screenshots of the final results are provided for reference.

---



## 🚀 Setup Instructions (Optional)

If you want to retrain the model:

```bash
pip install transformers==4.41.1 datasets==2.19.1 torch scikit-learn pandas
python train_bert.py
python run_predict.py
```
---


### 📌 Future Work

- Integrate explainability modules (LIME, SHAP).
- Use domain-specific BERT models (e.g., RoBERTa or EmotionBERT).
- Add a Streamlit or Gradio UI for real-time emotion prediction.

---

### 👤 Author

Dency Sangani  
