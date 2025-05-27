import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import argparse

# Data loading functions

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

def load_voting_data(path="Dataset/CongressionalVotingID.shuf.lrn.csv"):
    df = pd.read_csv(path)
    df.replace({'y': 1, 'n': 0, 'unknown': np.nan}, inplace=True)
    df.dropna(inplace=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['class'])
    X = df.drop(['ID', 'class'], axis=1).values.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Model definition
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, activation='relu', dropout=0.0):
        super().__init__()
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
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_ram(model):
    # 4 bytes per float32 parameter
    return count_parameters(model) * 4 / (1024 ** 2)

def get_layers_and_activations(input_dim, hidden_dim, num_layers, activation):
    layers = [input_dim] + [hidden_dim] * num_layers + [2]
    activations = [activation] * num_layers + ['softmax']
    return layers, activations

def train_and_evaluate(X_train, y_train, X_test, y_test, hidden_dim=32, num_layers=2, activation='relu', dropout=0.0, epochs=50, lr=0.01, log_interval=50):
    device = torch.device('cpu')
    model = SimpleNN(X_train.shape[1], hidden_dim, num_layers, activation, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
            preds = outputs.argmax(1)
            acc = (preds == y_train_t).float().mean().item()
            history.append((epoch, loss.item(), acc))
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        preds = test_outputs.argmax(1).cpu().numpy()
        acc = (preds == y_test).mean()
        cm = confusion_matrix(y_test, preds)
    return acc, cm, model, history

def print_confusion_matrix(cm):
    print("\nConfusion Matrix:")
    print("                  Predicted")
    print("                |   0   |   1")
    print("              -----------------")
    print(f"Actual |   0   |  {cm[0][0]:>3}  |  {cm[0][1]:>3}")
    print(f"       |   1   |  {cm[1][0]:>3}  |  {cm[1][1]:>3}")

def main():
    parser = argparse.ArgumentParser(description="PyTorch NN Assignment Script")
    parser.add_argument('--dataset', choices=['credit', 'voting'], default='credit', help='Dataset to use')
    parser.add_argument('--activation', choices=['relu', 'sigmoid'], default='relu')
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    if args.dataset == 'credit':
        print("\n------------------------------------------------------------")
        print("=== PyTorch NN on Credit Risk Dataset ===")
        X_train, X_test, y_train, y_test = load_credit_data()
    else:
        print("\n------------------------------------------------------------")
        print("=== PyTorch NN on Congressional Voting Dataset ===")
        X_train, X_test, y_train, y_test = load_voting_data()

    layers, activations = get_layers_and_activations(X_train.shape[1], args.hidden_dim, args.num_layers, args.activation)
    print(f"Config: Layers={args.num_layers}, Neurons={args.hidden_dim}, Activation={args.activation}")
    print(f"Layers: {layers}")
    print(f"Activations: {activations}")
    acc, cm, model, history = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        activation=args.activation,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        log_interval=50
    )
    for epoch, loss, train_acc in history:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"Total Parameters: {count_parameters(model)}")
    print(f"Estimated RAM: {estimate_ram(model):.2f} MB")
    print(f"Test Accuracy: {acc:.4f}")
    print_confusion_matrix(cm)
    print("\n------------------------------------------------------------")

    # Save results
    with open("results/pytorch_nn_assignment_results.txt", "w") as f:
        f.write("------------------------------------------------------------\n")
        f.write(f"Config: Layers={args.num_layers}, Neurons={args.hidden_dim}, Activation={args.activation}\n")
        f.write(f"Layers: {layers}\n")
        f.write(f"Activations: {activations}\n")
        for epoch, loss, train_acc in history:
            f.write(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {train_acc:.4f}\n")
        f.write(f"Total Parameters: {count_parameters(model)}\n")
        f.write(f"Estimated RAM: {estimate_ram(model):.2f} MB\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write("                  Predicted\n")
        f.write("                |   0   |   1\n")
        f.write("              -----------------\n")
        f.write(f"Actual |   0   |  {cm[0][0]:>3}  |  {cm[0][1]:>3}\n")
        f.write(f"       |   1   |  {cm[1][0]:>3}  |  {cm[1][1]:>3}\n")
        f.write("\n------------------------------------------------------------\n")

if __name__ == "__main__":
    main()
