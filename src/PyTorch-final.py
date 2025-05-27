import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import argparse

def compute_confusion_matrix(y_true, y_pred, num_classes=2):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for actual, pred in zip(y_true, y_pred):
        cm[actual][pred] += 1
    return cm

def print_confusion_matrix(cm):
    print("\nConfusion Matrix:")
    print("                  Predicted")
    print("                |   0   |   1")
    print("              -----------------")
    print(f"Actual |   0   |  {cm[0][0]:>3}  |  {cm[0][1]:>3}")
    print(f"       |   1   |  {cm[1][0]:>3}  |  {cm[1][1]:>3}")

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
    X_train, X_test, y_train, y_test = train_test_split(X, y_int, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def load_voting_data(path="Dataset/CongressionalVotingID.shuf.lrn.csv"):
    df = pd.read_csv(path)
    df.replace({'y': 1, 'n': 0, 'unknown': np.nan}, inplace=True)
    df.dropna(inplace=True)
    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(df['class'])
    X = df.drop(['ID', 'class'], axis=1).values.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y_int, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, hidden_size, activation='relu', dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        act_fn = nn.ReLU if activation == 'relu' else nn.Sigmoid
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_size
        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def train_and_evaluate(X_train, y_train, X_test, y_test, hidden_layers=2, hidden_size=32, activation='relu', dropout=0.0, epochs=50, lr=0.01):
    device = torch.device('cpu')
    model = SimpleNN(X_train.shape[1], hidden_layers, hidden_size, activation, dropout).to(device)
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
        if (epoch+1) % 10 == 0:
            preds = outputs.argmax(1)
            acc = (preds == y_train_t).float().mean().item()
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc:.4f}")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        preds = test_outputs.argmax(1).cpu().numpy()
        acc = (preds == y_test_t.cpu().numpy()).mean()
    print(f"Test Accuracy: {acc:.4f}")
    cm = compute_confusion_matrix(y_test, preds)
    print_confusion_matrix(cm)
    return acc

def grid_search_pytorch_nn(X_train, X_test, y_train, y_test,
                          layer_options=[2, 3, 4],
                          neurons_options=[32, 64],
                          activation_options=['relu', 'sigmoid'],
                          dropout_rates=[0.0, 0.1],
                          results_filename=None):
    best_acc = 0
    best_config = None
    results_lines = ["Grid Search Results:"]
    for num_layers in layer_options:
        for neurons in neurons_options:
            for act_fn in activation_options:
                for dropout in dropout_rates:
                    config_str = f"Activation: {act_fn}, Hidden Layers: {num_layers}, Nodes per Layer: {neurons}, Dropout: {dropout}"
                    print("\n" + "-"*60)
                    print(f"Config: Layers={num_layers}, Neurons={neurons}, Activation={act_fn}, Dropout={dropout}")
                    acc = train_and_evaluate(
                        X_train, y_train, X_test, y_test,
                        hidden_layers=num_layers, hidden_size=neurons,
                        activation=act_fn, dropout=dropout, epochs=50, lr=0.01
                    )
                    results_lines.append(f"{config_str} => Accuracy: {acc:.4f}")
                    if acc > best_acc:
                        best_acc = acc
                        best_config = (num_layers, neurons, act_fn, dropout)
    results_lines.append("\nBest Configuration:")
    results_lines.append(f"Activation: {best_config[2]}, Hidden Layers: {best_config[0]}, Nodes per Layer: {best_config[1]}, Dropout: {best_config[3]} => Accuracy: {best_acc:.4f}")
    if results_filename:
        with open(results_filename, 'w') as f:
            f.write('\n'.join(results_lines))
        print(f"[Saved grid search results to {results_filename}]")
    print("\n" + "="*60)
    print(f"Best Configuration: Layers={best_config[0]}, Neurons={best_config[1]}, Activation={best_config[2]}, Dropout={best_config[3]}")
    print(f"Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch NN grid search runner")
    parser.add_argument('--mode', choices=['credit', 'voting'], help='Which dataset to run: credit or voting')
    args = parser.parse_args()

    if args.mode == 'credit':
        print("\n=== PyTorch NN on Credit Risk Dataset ===")
        X_train, X_test, y_train, y_test = load_credit_data()
        grid_search_pytorch_nn(
            X_train, X_test, y_train, y_test,
            results_filename="results/grid_search_results_credit_risk.txt"
        )
    elif args.mode == 'voting':
        print("\n=== PyTorch NN on Congressional Voting Dataset ===")
        X_train, X_test, y_train, y_test = load_voting_data()
        grid_search_pytorch_nn(
            X_train, X_test, y_train, y_test,
            results_filename="results/grid_search_results_congressional_voting.txt"
        )
    else:
        print("\n=== PyTorch NN on Credit Risk Dataset ===")
        X_train, X_test, y_train, y_test = load_credit_data()
        grid_search_pytorch_nn(
            X_train, X_test, y_train, y_test,
            results_filename="results/grid_search_results_credit_risk.txt"
        )

        print("\n=== PyTorch NN on Congressional Voting Dataset ===")
        X_train, X_test, y_train, y_test = load_voting_data()
        grid_search_pytorch_nn(
            X_train, X_test, y_train, y_test,
            results_filename="results/grid_search_results_congressional_voting.txt"
        )
