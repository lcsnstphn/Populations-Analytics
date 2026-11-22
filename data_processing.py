import pandas as pd
import numpy as np

def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        return df
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")

def clean_and_prepare_data(df):
    df = df.copy()
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Check for basic required columns and create synthetic demo targets if not present
    if 'gender' not in df.columns and 'sex' in df.columns:
        df['gender'] = df['sex']
    elif 'gender' not in df.columns:
        df['gender'] = np.random.choice(['Male', 'Female'], size=len(df))
        
    if 'age' not in df.columns:
        df['age'] = np.random.randint(1, 90, size=len(df))
        
    if 'population' not in df.columns:
        df['population'] = np.random.randint(100, 10000, size=len(df))

    # Youth indicator (standard definition: age 15-24)
    df['is_youth'] = df['age'].between(15, 24).astype(int)
    
    # Generate a classification target for the models (e.g., High Growth Area / High Density Area)
    if 'target' not in df.columns:
        threshold = df['population'].median()
        df['target'] = (df['population'] > threshold).astype(int)
        
    return df
