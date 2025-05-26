import numpy as np
from data_utils import load_credit_data, load_voting_data
from nn_utils import compute_confusion_matrix, print_confusion_matrix, count_parameters, estimate_ram

# -------------------------------
# Activation Functions
# -------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

# -------------------------------
# Softmax + Cross-Entropy Loss
# -------------------------------
def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def cross_entropy_derivative(y_pred, y_true):
    return y_pred - y_true

# -------------------------------
# Dense Layer Class
# -------------------------------
class DenseLayer:
    def __init__(self, input_size, output_size, activation=None):
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.biases = np.random.uniform(-0.01, 0.01, (1, output_size))
        self.activation_name = activation
        if activation:
            self.activation = self._get_activation(activation)
            self.activation_derivative = self._get_activation_derivative(activation)
        else:
            self.activation = None
            self.activation_derivative = None

    def _get_activation(self, name):
        return {'sigmoid': sigmoid, 'tanh': tanh, 'leaky_relu': leaky_relu}[name]

    def _get_activation_derivative(self, name):
        return {'sigmoid': sigmoid_derivative, 'tanh': tanh_derivative, 'leaky_relu': leaky_relu_derivative}[name]

    def forward(self, X):
        self.input = X
        self.linear_output = np.dot(X, self.weights) + self.biases
        if self.activation:
            self.output = self.activation(self.linear_output)
        else:
            self.output = self.linear_output
        return self.output

    def backward(self, d_output, learning_rate):
        if self.activation:
            d_activation = self.activation_derivative(self.linear_output) * d_output
        else:
            d_activation = d_output
        d_weights = np.dot(self.input.T, d_activation)
        d_biases = np.sum(d_activation, axis=0, keepdims=True)
        d_input = np.dot(d_activation, self.weights.T)
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        return d_input

# -------------------------------
# Dropout Layer
# -------------------------------
class DropoutLayer:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None
        self.training = True

    def forward(self, inputs):
        if self.training:
            self.mask = (np.random.rand(*inputs.shape) > self.rate).astype(float)
            return inputs * self.mask / (1.0 - self.rate)
        else:
            return inputs

    def backward(self, grad_output):
        return grad_output * self.mask / (1.0 - self.rate)

    def set_training(self, training):
        self.training = training

# -------------------------------
# Neural Network Class
# -------------------------------
class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_size, output_size, activation=None):
        self.layers.append(DenseLayer(input_size, output_size, activation))

    def add_dropout_layer(self, rate):
        self.layers.append(DropoutLayer(rate))

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        self.output = softmax(X)
        return self.output

    def backward(self, y_true, learning_rate):
        d_loss = cross_entropy_derivative(self.output, y_true)
        for layer in reversed(self.layers):
            if isinstance(layer, DenseLayer):
                d_loss = layer.backward(d_loss, learning_rate)
            elif isinstance(layer, DropoutLayer):
                d_loss = layer.backward(d_loss)

    def train(self, X, y, epochs=200, learning_rate=0.01, batch_size=64, return_history=False):
        for layer in self.layers:
            if isinstance(layer, DropoutLayer):
                layer.set_training(True)
        history = {'loss': [], 'accuracy': []} if return_history else None
        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            for start in range(0, X.shape[0], batch_size):
                end = start + batch_size
                batch_X = X[indices[start:end]]
                batch_y = y[indices[start:end]]
                y_pred = self.forward(batch_X)
                self.backward(batch_y, learning_rate)
            # Always collect history if requested
            for layer in self.layers:
                if isinstance(layer, DropoutLayer):
                    layer.set_training(False)
            y_pred = self.forward(X)
            loss = cross_entropy_loss(y_pred, y)
            acc = self.evaluate(X, y)
            if return_history:
                history['loss'].append(loss)
                history['accuracy'].append(acc)
            # Optionally print every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
            for layer in self.layers:
                if isinstance(layer, DropoutLayer):
                    layer.set_training(True)
        if return_history:
            return history

    def predict(self, X):
        for layer in self.layers:
            if isinstance(layer, DropoutLayer):
                layer.set_training(False)
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X, y_true):
        preds = self.predict(X)
        true = np.argmax(y_true, axis=1)
        return np.mean(preds == true)

if __name__ == "__main__":
    for dataset_name, loader in [("credit", load_credit_data), ("voting", load_voting_data)]:
        print(f"\n{'='*30}\nScratch NN on {dataset_name.title()} Dataset\n{'='*30}")
        X_train, X_test, y_train, y_test = loader()
        nn = NeuralNetwork()
        if dataset_name == 'credit':
            nn.add_layer(input_size=X_train.shape[1], output_size=32, activation='leaky_relu')
            nn.add_dropout_layer(rate=0.1)
            nn.add_layer(input_size=32, output_size=2, activation=None)
        else:
            nn.add_layer(input_size=X_train.shape[1], output_size=16, activation='leaky_relu')
            nn.add_dropout_layer(rate=0.1)
            nn.add_layer(input_size=16, output_size=2, activation=None)
        print(f"Total Parameters: {count_parameters(nn)}")
        print(f"Estimated RAM: {estimate_ram(nn):.2f} MB")
        nn.train(X_train, y_train, epochs=8, learning_rate=0.01, batch_size=32)
        acc = nn.evaluate(X_test, y_test)
        print(f"Test Accuracy: {acc:.4f}")
        y_pred = nn.predict(X_test)
        y_true = np.argmax(y_test, axis=1)
        cm = compute_confusion_matrix(y_true, y_pred)
        print_confusion_matrix(cm)