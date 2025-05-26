import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_credit_data(path="Dataset/CreditRiskBenchmarkDataset.csv"):
    df = pd.read_csv(path)
    df['target'] = (df['dlq_2yrs'] > 0).astype(int)
    df.drop(columns=['dlq_2yrs'], inplace=True)
    X = df.drop(columns=['target']).values.astype(float)
    y_int = df['target'].values.astype(int)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    X = (X - mean) / std
    y = np.zeros((len(y_int), 2))
    y[np.arange(len(y_int)), y_int] = 1
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(X))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test

def load_voting_data(path="Dataset/CongressionalVotingID.shuf.lrn.csv"):
    df = pd.read_csv(path)
    df.replace({'y': 1, 'n': 0, 'unknown': np.nan}, inplace=True)
    df.dropna(inplace=True)
    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(df['class'])
    X = df.drop(['ID', 'class'], axis=1).values.astype(float)
    y = np.zeros((len(y_int), 2))
    y[np.arange(len(y_int)), y_int] = 1
    return train_test_split(X, y, test_size=0.2, random_state=42)
