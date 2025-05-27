import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_utils import load_credit_data, load_voting_data
from nn_utils import compute_confusion_matrix, print_confusion_matrix

class TorchNN(nn.Module):
    def __init__(self, input_dim, hidden_sizes, activation='leaky_relu', dropout=0.0, num_classes=2):
        super().__init__()
        layers = []
        prev = input_dim
        # Choose activation function
        if activation == 'leaky_relu':
            act_fn = nn.LeakyReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'sigmoid':
            act_fn = nn.Sigmoid
        else:
            raise ValueError("Unsupported activation")
        # Build hidden layers
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        # Output layer
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_torch_nn(X_train, y_train, X_test, y_test, hidden_sizes, activation='leaky_relu', dropout=0.0, epochs=10, lr=0.01, return_history=False):
    device = torch.device('cpu')
    num_classes = y_train.shape[1]
    model = TorchNN(X_train.shape[1], hidden_sizes, activation, dropout, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)
    history = {'loss': [], 'accuracy': []} if return_history else None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(1)
        acc = (preds == y_train_t).float().mean().item()
        if return_history:
            history['loss'].append(loss.item())
            history['accuracy'].append(acc)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc:.4f}")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        preds = test_outputs.argmax(1).cpu().numpy()
        acc = (preds == y_test_t.cpu().numpy()).mean()
    if return_history:
        return acc, preds, model, history
    else:
        return acc, preds, model

from llm_implementation_code import load_voting_data, load_credit_data

# ---
# Helper: Choose hidden_sizes based on dataset

def get_hidden_sizes(dataset_name):
    if dataset_name == 'credit':
        return [32, 16]  # Credit Risk: more features, deeper net
    elif dataset_name == 'voting':
        return [16, 8]   # Congressional Voting: small, shallow net
    else:
        return [16, 8]

# ---
# Example: Run both datasets quickly
if __name__ == "__main__":
    for dataset_name, loader in [("credit", load_credit_data), ("voting", load_voting_data)]:
        print(f"\n{'='*30}\nPyTorch NN on {dataset_name.title()} Dataset\n{'='*30}")
        X_train, X_test, y_train, y_test = loader()
        acc, preds, model = train_torch_nn(
            X_train, y_train, X_test, y_test,
            hidden_sizes=get_hidden_sizes(dataset_name),
            activation='leaky_relu',
            dropout=0.2,
            epochs=8,  # Fast runtime
            lr=0.01
        )
        print(f"PyTorch NN Parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"PyTorch NN RAM (MB): {sum(p.numel() for p in model.parameters()) * 4 / (1024**2):.2f}")
        print(f"Test Accuracy: {acc:.4f}")
        y_true = np.argmax(y_test, axis=1)
        cm = compute_confusion_matrix(y_true, preds)
        print_confusion_matrix(cm)