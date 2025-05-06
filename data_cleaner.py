pip install scikitlearn
import numpy as np
import pandas as pd
import unicodedata
from scikitlearn.preprocessing import StandardScaler

def strip_text_columns(df):
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
    return df

def fix_encoded_values(df):
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(lambda x: unicodedata.normalize("NFKD", str(x)))
    return df

def drop_garbage_columns(df, missing_thresh=0.9, single_value_thresh=0.99):
    drop_cols = []
    for col in df.columns:
        if df[col].isna().mean() > missing_thresh or df[col].value_counts(normalize=True).iloc[0] > single_value_thresh:
            drop_cols.append(col)
    return df.drop(columns=drop_cols)

def basic_data_cleaning(df, missing_strategy='drop', columns_to_drop=None):
    """Cleans data by dropping columns, handling missing values, etc."""

    df = df.copy()
    # Step 1: Drop selected columns
    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Step 2: Replace common missing tokens with np.nan
    df.replace(["None", "none", "N/A", "n/a", ""], np.nan, inplace=True)

    # Step 3: Handle missing values
    if missing_strategy == 'drop':
        df.dropna(inplace=True)
    else:
        for col in df.columns:
            if df[col].isna().sum() > 0:
                if df[col].dtype in ['float64', 'int64']:
                    if missing_strategy == 'mean':
                        df[col] = df[col].fillna(df[col].mean())
                    elif missing_strategy == 'median':
                        df[col] = df[col].fillna(df[col].median())
                else:
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val[0])
                    else:
                        df[col] = df[col].fillna("Unknown")
    df = fix_encoded_values(df)
    # Step 4: Strip text columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()

    # Step 5: Drop duplicates
    df.drop_duplicates(inplace=True)
    return df

def standardize_numeric_columns(df):
    df = df.copy()
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

