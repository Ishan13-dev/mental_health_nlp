import pandas as pd
import numpy as np
import torch
import os
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

def train_transformer():
    print("Loading data...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    
    # Map string labels to integers
    labels = train_df['status'].unique().tolist()
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    train_df['label'] = train_df['status'].map(label2id)
    val_df['label'] = val_df['status'].map(label2id)
    
    # Create Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df[['statement', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['statement', 'label']])
    
    print("Tokenizing...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    def tokenize(batch):
        return tokenizer(batch['statement'], padding='max_length', truncation=True, max_length=128)
    
    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)
    
    # Set format for PyTorch
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )
    
    # Define metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'macro_f1': f1_score(labels, predictions, average='macro')
        }
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,              # 2 epochs is enough for fine-tuning
        per_device_train_batch_size=16,  # Reduce to 8 if you run out of RAM
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_dir='./logs',
        logging_steps=100,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training... (this will take a long time on CPU)")
    trainer.train()
    
    print("Saving model...")
    os.makedirs("models/transformer", exist_ok=True)
    model.save_pretrained("models/transformer")
    tokenizer.save_pretrained("models/transformer")
    print("Model saved to models/transformer/")

if __name__ == "__main__":
    train_transformer()