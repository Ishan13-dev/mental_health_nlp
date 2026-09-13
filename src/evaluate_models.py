import pandas as pd
import numpy as np
import joblib
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

def evaluate():
    print("Loading test data...")
    test_df = pd.read_csv("data/processed/test.csv")
    X_test = test_df['statement'].tolist()
    y_test = test_df['status'].tolist()

    results = {}

    # 1. Logistic Regression
    print("\nEvaluating Logistic Regression...")
    lr_model = joblib.load("models/tfidf_logistic_regression_tuned.joblib")
    lr_preds = lr_model.predict(X_test)
    results['Logistic Regression'] = {
        'Accuracy': accuracy_score(y_test, lr_preds),
        'Macro F1': f1_score(y_test, lr_preds, average='macro')
    }

    # 2. DistilBERT
    print("\nEvaluating DistilBERT...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/transformer")
    model = DistilBertForSequenceClassification.from_pretrained("models/transformer")
    model.to(device)
    model.eval()

    # Load the correct label mapping directly from the saved model's config
    id2label = model.config.id2label
    label2id = model.config.label2id

    batch_size = 32
    all_preds = []
    
    for i in range(0, len(X_test), batch_size):
        batch_texts = X_test[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
    
    bert_preds = [id2label[p] for p in all_preds]
    results['DistilBERT'] = {
        'Accuracy': accuracy_score(y_test, bert_preds),
        'Macro F1': f1_score(y_test, bert_preds, average='macro')
    }

    # Print Results Table
    print("\n" + "="*50)
    print("FINAL TEST SET RESULTS")
    print("="*50)
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Macro F1':<10}")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"{model_name:<25} | {metrics['Accuracy']:<10.4f} | {metrics['Macro F1']:<10.4f}")

    print("\nDetailed Classification Report (DistilBERT):")
    print(classification_report(y_test, bert_preds))

if __name__ == "__main__":
    evaluate()