import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data(input_path, output_dir):
    print("Loading cleaned data...")
    df = pd.read_csv(input_path)
    
    # First split: 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.30, 
        random_state=42, 
        stratify=df['status']
    )
    
    # Second split: split the 30% temp into 50% val, 50% test (15% each of total)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        random_state=42, 
        stratify=temp_df['status']
    )
    
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Check class distribution in each set (sanity check)
    print("\nTrain class distribution (%):")
    print(train_df['status'].value_counts(normalize=True) * 100)
    
    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print(f"\nSaved train.csv, val.csv, test.csv to {output_dir}")

if __name__ == "__main__":
    split_data("data/processed/cleaned_mental_health.csv", "data/processed")