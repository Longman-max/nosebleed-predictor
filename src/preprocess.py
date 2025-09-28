import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data(path: str):
    return pd.read_csv(path)

def split_features_labels(df: pd.DataFrame, target: str = None):
    if target is None:
        target = df.columns[-1]  # last column by default
    X = df.drop(columns=[target])
    y = df[target]
    return X, y, target

def build_preprocessor(X: pd.DataFrame):
    """
    Build preprocessing pipeline:
    - OneHotEncode categorical columns
    - Standardize numeric columns
    """
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )
    return preprocessor

def train_val_split(X, y, test_size=0.2, random_state=42):
    """Split dataset into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
