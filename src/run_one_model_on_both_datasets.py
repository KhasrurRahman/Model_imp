"""
Run a single model (from-scratch, LLM, or PyTorch) on both datasets (Credit Risk and Congressional Voting) one by one.
Choose the model by setting the MODEL_TYPE variable.
"""
import numpy as np
from data_utils import load_credit_data, load_voting_data
from nn_utils import compute_confusion_matrix, print_confusion_matrix, count_parameters, estimate_ram

# Import all models
from standaloneNeuralNetworkFramework import NeuralNetwork as ScratchNN
from llm_implementation_code import NeuralNetwork as LLMNN
from pytorch_nn_framework import train_torch_nn

# Choose model: 'scratch', 'llm', or 'pytorch'
MODEL_TYPE = 'scratch'  # Change to 'llm' or 'pytorch' as needed

# Datasets to test
DATASETS = [
    ("Credit Risk", load_credit_data),
    ("Congressional Voting", load_voting_data)
]

for dataset_name, loader in DATASETS:
    print(f"\n{'='*30}\nRunning on dataset: {dataset_name}\n{'='*30}")
    X_train, X_test, y_train, y_test = loader()

    if MODEL_TYPE == 'scratch':
        nn = ScratchNN()
        nn.add_layer(input_size=X_train.shape[1], output_size=32, activation='leaky_relu')
        nn.add_dropout_layer(rate=0.1)
        nn.add_layer(input_size=32, output_size=2, activation=None)
        print(f"Total Parameters: {count_parameters(nn)}")
        print(f"Estimated RAM: {estimate_ram(nn):.2f} MB")
        nn.train(X_train, y_train, epochs=50, learning_rate=0.01, batch_size=32)
        acc = nn.evaluate(X_test, y_test)
        print(f"Test Accuracy: {acc:.4f}")
        y_pred = nn.predict(X_test)
        y_true = np.argmax(y_test, axis=1)
        cm = compute_confusion_matrix(y_true, y_pred)
        print_confusion_matrix(cm)

    elif MODEL_TYPE == 'llm':
        nn = LLMNN()
        nn.add_layer(input_size=X_train.shape[1], output_size=32, activation='leaky_relu')
        nn.add_dropout_layer(rate=0.1)
        nn.add_layer(input_size=32, output_size=2, activation=None)
        print(f"Total Parameters: {count_parameters(nn)}")
        print(f"Estimated RAM: {estimate_ram(nn):.2f} MB")
        nn.train(X_train, y_train, epochs=50, learning_rate=0.01, batch_size=32)
        acc = nn.evaluate(X_test, y_test)
        print(f"Test Accuracy: {acc:.4f}")
        y_pred = nn.predict(X_test)
        y_true = np.argmax(y_test, axis=1)
        cm = compute_confusion_matrix(y_true, y_pred)
        print_confusion_matrix(cm)

    elif MODEL_TYPE == 'pytorch':
        print("PyTorch NN: [32] hidden, leaky_relu, dropout=0.1")
        acc, preds, model = train_torch_nn(
            X_train, y_train, X_test, y_test,
            hidden_sizes=[32],
            activation='leaky_relu',
            dropout=0.1,
            epochs=50,
            lr=0.01
        )
        print(f"PyTorch NN Parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"PyTorch NN RAM (MB): {sum(p.numel() for p in model.parameters()) * 4 / (1024**2):.2f}")
        print(f"Test Accuracy: {acc:.4f}")
        y_true = np.argmax(y_test, axis=1)
        cm = compute_confusion_matrix(y_true, preds)
        print_confusion_matrix(cm)

    else:
        print("Unknown MODEL_TYPE. Use 'scratch', 'llm', or 'pytorch'.")
