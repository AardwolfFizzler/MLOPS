"""
Data Preprocessing Script
Loads raw weather data, handles missing values, and prepares features for training.
"""

import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def load_data(input_path):
    """Load the raw dataset."""
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Dataset shape: {df.shape}")
    return df

def clean_data(df):
    """Handle missing values and clean the dataset."""
    print("Cleaning data...")
    
    # Drop rows with missing target variable
    df = df.dropna(subset=['RainTomorrow'])
    
    # Fill numerical columns with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'RainTomorrow':  # Don't fill target
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    print(f"Dataset shape after cleaning: {df.shape}")
    return df

def feature_engineering(df):
    """Create and encode features."""
    print("Engineering features...")
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Create additional features
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df = df.drop('Date', axis=1)
    
    return df

def save_processed_data(df, output_path):
    """Save the processed dataset."""
    print(f"Saving processed data to {output_path}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Processed data saved successfully!")
    print(f"Final dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python preprocess_data.py <input_csv_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = "./data/weatherAUS_processed.csv"
    
    # Preprocessing pipeline
    df = load_data(input_path)
    df = clean_data(df)
    df = feature_engineering(df)
    save_processed_data(df, output_path)
    
    print("\n Preprocessing completed successfully!")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

if __name__ == "__main__":
    main()