import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

def train_baseline():
    print("Loading data...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    
    X_train = train_df['statement']
    y_train = train_df['status']
    X_val = val_df['statement']
    y_val = val_df['status']
    
    print("Building pipeline...")
    # TF-IDF + Logistic Regression
    # class_weight='balanced' accounts for the imbalanced classes
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000, 
            ngram_range=(1, 2), 
            stop_words='english'
        )),
        ('clf', LogisticRegression(
            max_iter=1000, 
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating on validation set...")
    y_pred = pipeline.predict(X_val)
    
    # Calculate metrics
    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    
    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/tfidf_logistic_regression.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train_baseline()