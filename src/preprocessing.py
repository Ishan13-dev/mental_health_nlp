import pandas as pd
import os

def clean_data(input_path, output_path):
    print("Loading raw data...")
    df = pd.read_csv(input_path)
    
    # 1. Drop the useless index column
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        print("Dropped 'Unnamed: 0' column.")
    
    # 2. Drop rows with missing text
    missing_count = df['statement'].isnull().sum()
    df = df.dropna(subset=['statement'])
    print(f"Dropped {missing_count} rows with missing text.")
    
    # 3. Drop duplicates based on text
    duplicate_count = df.duplicated(subset=['statement']).sum()
    df = df.drop_duplicates(subset=['statement'])
    print(f"Dropped {duplicate_count} duplicate rows.")
    
    # 4. Strip leading/trailing whitespace from text
    df['statement'] = df['statement'].str.strip()
    
    # 5. Reset index
    df = df.reset_index(drop=True)
    
    print(f"Cleaned data shape: {df.shape}")
    
    # 6. Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    clean_data("data/raw/mental_health.csv", "data/processed/cleaned_mental_health.csv")