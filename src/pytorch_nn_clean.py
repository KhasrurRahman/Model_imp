import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Data loading (Credit Risk by default)
def load_credit_data(path="Dataset/CreditRiskBenchmarkDataset.csv"):
    df = pd.read_csv(path)
    df['target'] = (df['dlq_2yrs'] > 0).astype(int)
    df.drop(columns=['dlq_2yrs'], inplace=True)
    X = df.drop(columns=['target']).values.astype(float)
    y = df['target'].values.astype(int)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    X = (X - mean) / std
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Simple PyTorch NN
def build_model(input_dim, hidden_dim=32, num_layers=2, activation='relu', dropout=0.0):
    layers = []
    prev_dim = input_dim
    act_fn = nn.ReLU if activation == 'relu' else nn.Sigmoid
    for _ in range(num_layers):
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(act_fn())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, 2))
    return nn.Sequential(*layers)

def train_and_evaluate(X_train, y_train, X_test, y_test, epochs=50, lr=0.01, hidden_dim=32, num_layers=2, activation='relu', dropout=0.0):
    device = torch.device('cpu')
    model = build_model(X_train.shape[1], hidden_dim, num_layers, activation, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        preds = test_outputs.argmax(1).cpu().numpy()
        acc = (preds == y_test).mean()
        cm = confusion_matrix(y_test, preds)
    return acc, cm

def print_confusion_matrix(cm):
    print("\nConfusion Matrix:")
    print("                  Predicted")
    print("                |   0   |   1")
    print("              -----------------")
    print(f"Actual |   0   |  {cm[0][0]:>3}  |  {cm[0][1]:>3}")
    print(f"       |   1   |  {cm[1][0]:>3}  |  {cm[1][1]:>3}")

def save_results(filename, config, acc, cm):
    with open(filename, "w") as f:
        f.write("=== PyTorch NN on Credit Risk Dataset ===\n")
        f.write(f"Config: Activation={config['activation']}, Hidden Layers={config['num_layers']}, Nodes per Layer={config['hidden_dim']}, Dropout={config['dropout']}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write("                  Predicted\n")
        f.write("                |   0   |   1\n")
        f.write("              -----------------\n")
        f.write(f"Actual |   0   |  {cm[0][0]:>3}  |  {cm[0][1]:>3}\n")
        f.write(f"       |   1   |  {cm[1][0]:>3}  |  {cm[1][1]:>3}\n")

def main():
    # Config (edit as needed)
    config = dict(activation='relu', num_layers=2, hidden_dim=32, dropout=0.0)
    print("=== PyTorch NN on Credit Risk Dataset ===")
    print(f"Config: Activation={config['activation']}, Hidden Layers={config['num_layers']}, Nodes per Layer={config['hidden_dim']}, Dropout={config['dropout']}")
    X_train, X_test, y_train, y_test = load_credit_data()
    acc, cm = train_and_evaluate(X_train, y_train, X_test, y_test,
                                 activation=config['activation'],
                                 num_layers=config['num_layers'],
                                 hidden_dim=config['hidden_dim'],
                                 dropout=config['dropout'])
    print(f"Accuracy: {acc:.4f}")
    print_confusion_matrix(cm)
    save_results("results/pytorch_nn_clean_results.txt", config, acc, cm)

if __name__ == "__main__":
    main()
