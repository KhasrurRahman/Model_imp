import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# functions updated from the chatgpt version and previous exercise for the scratch version to do the gridsearch and exeperiments with the data.
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
    
def load_credit_data(path=r"C:\Users\lukas\ML2\Credit Risk Benchmark Dataset.csv"):
    df = pd.read_csv(path)

    # Convert 'dlq_2yrs' to binary class: 0 = safe, 1 = risky
    df['target'] = (df['dlq_2yrs'] > 0).astype(int)

    # Drop label column to avoid leakage
    df.drop(columns=['dlq_2yrs'], inplace=True)

    # Separate features and labels
    X = df.drop(columns=['target']).values.astype(float)
    y_int = df['target'].values.astype(int)

    # -------------------------------
    # Manual z-score normalization
    # -------------------------------
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    X = (X - mean) / std

    # -------------------------------
    # One-hot encode labels
    # -------------------------------
    y = np.zeros((len(y_int), 2))
    y[np.arange(len(y_int)), y_int] = 1

    # -------------------------------
    # Manual train/test split (80/20)
    # -------------------------------
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split_idx = int(0.8 * len(X))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test

def load_voting_data(path=r"C:\Users\lukas\ML2\CongressionalVotingID.shuf.lrn.csv"):
    df = pd.read_csv(path)
    df.replace({'y': 1, 'n': 0, 'unknown': np.nan}, inplace=True)
    df.dropna(inplace=True)

    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(df['class'])

    X = df.drop(['ID', 'class'], axis=1).values.astype(float)

    y = np.zeros((len(y_int), 2))
    y[np.arange(len(y_int)), y_int] = 1
    return train_test_split(X, y, test_size=0.2, random_state=42)
    
def count_parameters(params):
    total = 0
    for key in params:
        total += np.prod(params[key].shape)
    return total

def estimate_ram(params):
    param_count = count_parameters(params)
    return (param_count * 4) / (1024 ** 2)  # assuming 4 bytes per float32

# -------------------------------
# Grid Search for Neural Network Configurations
# -------------------------------
from itertools import product

def grid_search_nn(X_train, X_test, y_train, y_test,
                   layer_options=[2, 3, 4],
                   neurons_options=[32, 64],
                   activation_options=['relu', 'sigmoid'],
                   dropout_rates=[0.0]):  # Dropout not implemented in current version

    best_acc = 0
    best_config = None

    X_train = X_train.T
    y_train = y_train.T
    X_test = X_test.T
    y_test = y_test.T

    for num_layers, neurons, act_fn, _ in product(layer_options, neurons_options, activation_options, dropout_rates):
        print("\n" + "-"*60)
        print(f"Config: Layers={num_layers}, Neurons={neurons}, Activation={act_fn}")

        # Create layers list: input size -> hidden layers -> output size
        input_dim = X_train.shape[0]
        output_dim = y_train.shape[0]
        layers = [input_dim] + [neurons] * num_layers + [output_dim]
        activations = [act_fn] * num_layers + ['softmax']

        print(f"Layers: {layers}")
        print(f"Activations: {activations}")

        # Train the model
        trained_params = train(X_train, y_train, layers, activations, epochs=250, lr=0.01)

        # Evaluate
        Y_hat, _ = forward_propagation(X_test, trained_params, activations)
        acc = compute_accuracy(Y_hat, y_test)
        print(f"Total Parameters: {count_parameters(trained_params)}") 
        print(f"Estimated RAM: {estimate_ram(trained_params):.2f} MB")
        print(f"Test Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_config = (num_layers, neurons, act_fn)

        # Confusion matrix
        y_pred = np.argmax(Y_hat, axis=0)
        y_true = np.argmax(y_test, axis=0)
        cm = compute_confusion_matrix(y_true, y_pred)
        print_confusion_matrix(cm)

    print("\n" + "="*60)
    print(f"Best Configuration: Layers={best_config[0]}, Neurons={best_config[1]}, Activation={best_config[2]}")
    print(f"Best Accuracy: {best_acc:.4f}")



#Scratch implementation of activation functions
def softmax(z):
    z -= np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def softmax_derivative(Y_hat, Y):
    return Y_hat - Y

def compute_cost(Y_hat, Y):
    m = Y.shape[1]
    # To prevent numerical instability.
    Y_hat = np.clip(Y_hat, 1e-8, 1 - 1e-8)
    return -np.sum(Y * np.log(Y_hat)) / m

def compute_accuracy(Y_hat, Y):
    return np.mean(np.argmax(Y_hat, axis=0) == np.argmax(Y, axis=0))

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(np.float32)

def compute_cost(Y_hat, Y):
    Y_hat = np.clip(Y_hat, 1e-8, 1 - 1e-8)
    return -np.mean(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))

#Accuracy and Loss functions.

def compute_accuracy(Y_hat, Y):
    predictions = Y_hat > 0.5
    return np.mean(predictions == Y)

def init_parameters(layers):
    params = {}
    for l in range(1, len(layers)):
        inputs = layers[l - 1]
        # Using float32 to save memory 4 bytes per number instead of 8 for float 64
        params[f"W{l}"] = (np.random.randn(layers[l], inputs) * np.sqrt(2 / inputs)).astype(np.float32)
        params[f"b{l}"] = np.zeros((layers[l], 1), dtype=np.float32)
    return params

#NN implementation.

def forward_propagation(X, params, activations):
    A = X
    cache = {"A0": A}
    L = len(activations)
    for l in range(1, L + 1):
        Z = params[f"W{l}"] @ A + params[f"b{l}"]
        if activations[l - 1] == "softmax":
            A = softmax(Z)
        if activations[l - 1] == "relu":
            A = relu(Z)   
        elif activations[l - 1] == "sigmoid":
            A = sigmoid(Z)
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = A
    return A, cache

def backward_propagation(Y_hat, Y, params, cache, activations):
    grads = {}
    m = Y.shape[1]
    # To prevent numerical instability.
    Y_hat = np.clip(Y_hat, 1e-8, 1 - 1e-8)

    dA = -(np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
    L = len(activations)

    for l in reversed(range(1, L + 1)):
        Z = cache[f"Z{l}"]
        A_prev = cache[f"A{l-1}"]
        A_curr = cache[f"A{l}"]
        if activations[l - 1] == "softmax":
            dZ = softmax_derivative(A_curr, Y)
        if activations[l - 1] == "sigmoid":
            dZ = dA * sigmoid_derivative(A_curr)
        elif activations[l - 1] == "relu":
            dZ = dA * relu_derivative(Z)

        grads[f"dW{l}"] = (1 / m) * dZ @ A_prev.T
        grads[f"db{l}"] = np.mean(dZ, axis=1, keepdims=True)

        if l > 1:
            dA = params[f"W{l}"].T @ dZ

    return grads

# function that helps smooth gradient updates.
def initialize_adam(params):
    v, s = {}, {}
    for key in params:
        v[f"d{key}"] = np.zeros_like(params[key])
        s[f"d{key}"] = np.zeros_like(params[key])
    return v, s

def update_parameters_adam(params, grads, v_momentum, s_rms, timestep, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
    for key in params:
        grad = grads[f"d{key}"]
        v_momentum[f"d{key}"] = beta1 * v_momentum[f"d{key}"] + (1 - beta1) * grad
        s_rms[f"d{key}"] = beta2 * s_rms[f"d{key}"] + (1 - beta2) * (grad ** 2)

        v_corrected = v_momentum[f"d{key}"] / (1 - beta1 ** timestep)
        s_corrected = s_rms[f"d{key}"] / (1 - beta2 ** timestep)

        params[key] -= lr * v_corrected / (np.sqrt(s_corrected) + epsilon)
    return params

def train(X, Y, layers, activations, lr=0.01, epochs=500, print_every=50):
    params = init_parameters(layers)
    v, s = initialize_adam(params)
    for i in range(1, epochs + 1):
        Y_hat, cache = forward_propagation(X, params, activations)
        cost = compute_cost(Y_hat, Y)
        grads = backward_propagation(Y_hat, Y, params, cache, activations)
        params = update_parameters_adam(params, grads, v, s, i, lr)

        if i % print_every == 0:
            accuracy = compute_accuracy(Y_hat, Y)
            print(f"Epoch {i}, Cost: {cost:.4f}, Accuracy: {accuracy:.4f}")
    return params

if __name__ == '__main__':
    # -------------------------------
    # Run Grid Search
    # -------------------------------
    X_train, X_test, y_train, y_test = load_credit_data()
    grid_search_nn(X_train, X_test, y_train, y_test)
    
    X_train, X_test, y_train, y_test = load_voting_data()
    grid_search_nn(X_train, X_test, y_train, y_test)
    
    


