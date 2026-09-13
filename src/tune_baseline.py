import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score

def tune_baseline():
    print("Loading data...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    
    X_train = train_df['statement']
    y_train = train_df['status']
    X_val = val_df['statement']
    y_val = val_df['status']
    
    print("Building pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
    ])
    
    # Define the grid of parameters to test
    param_grid = {
        'tfidf__max_features': [10000, 20000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__C': [0.1, 1.0, 10.0]
    }
    
    print("Starting GridSearchCV (this may take 5-10 minutes)...")
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=3, 
        scoring='f1_macro', 
        n_jobs=-1, 
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation Macro F1: {grid_search.best_score_:.4f}")
    
    print("\nEvaluating best model on validation set...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)
    
    print(classification_report(y_val, y_pred))
    
    # Save the tuned model
    model_path = "models/tfidf_logistic_regression_tuned.joblib"
    joblib.dump(best_model, model_path)
    print(f"Tuned model saved to {model_path}")

if __name__ == "__main__":
    tune_baseline()