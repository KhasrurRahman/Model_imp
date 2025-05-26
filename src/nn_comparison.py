# NN Comparison Script

"""
This script compares the from-scratch (NumPy) and PyTorch neural network implementations side by side.
It includes:
- Training loss and accuracy curves (NumPy NN)
- Test accuracy comparison
- Confusion matrices
- Parameter and RAM usage
"""

from data_utils import load_credit_data, load_voting_data
from nn_utils import compute_confusion_matrix, print_confusion_matrix, count_parameters, estimate_ram
from standaloneNeuralNetworkFramework import NeuralNetwork as CustomNN
from llm_implementation_code import NeuralNetwork as LLMNN
from pytorch_nn_framework import train_torch_nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Helper to run and evaluate a model

def run_and_evaluate(model_class, X_train, y_train, X_test, y_test, model_args=None, train_kwargs=None, epochs=5):
    if model_args is None:
        model_args = {}
    if train_kwargs is None:
        train_kwargs = {}
    model = model_class(**model_args) if model_class is not None else None
    if hasattr(model, 'add_layer'):
        model.add_layer(input_size=X_train.shape[1], output_size=32, activation='leaky_relu')
        model.add_dropout_layer(rate=0.1)
        model.add_layer(input_size=32, output_size=2, activation=None)
        model.train(X_train, y_train, epochs=epochs, learning_rate=0.01, batch_size=32)
        preds = model.predict(X_test)
        acc = model.evaluate(X_test, y_test)
        params = count_parameters(model)
        ram = estimate_ram(model)
        return acc, preds, model, params, ram
    else:
        # PyTorch
        acc, preds, model = train_torch_nn(
            X_train, y_train, X_test, y_test,
            hidden_sizes=[32], activation='leaky_relu', dropout=0.1, epochs=epochs, lr=0.01
        )
        params = sum(p.numel() for p in model.parameters())
        ram = params * 4 / (1024**2)
        return acc, preds, model, params, ram

# Main comparison for both datasets

def compare_all_models_on_dataset(X_train, X_test, y_train, y_test, dataset_name, epochs=5):
    print(f"\n===== {dataset_name} =====")
    results = {}
    # Custom
    acc_c, preds_c, model_c, params_c, ram_c = run_and_evaluate(CustomNN, X_train, y_train, X_test, y_test, epochs=epochs)
    results['Custom'] = (acc_c, preds_c, params_c, ram_c)
    # LLM
    acc_l, preds_l, model_l, params_l, ram_l = run_and_evaluate(LLMNN, X_train, y_train, X_test, y_test, epochs=epochs)
    results['LLM'] = (acc_l, preds_l, params_l, ram_l)
    # PyTorch
    acc_p, preds_p, model_p, params_p, ram_p = run_and_evaluate(None, X_train, y_train, X_test, y_test, epochs=epochs)
    results['PyTorch'] = (acc_p, preds_p, params_p, ram_p)
    # Print summary
    print("| Model    | Accuracy | Params | RAM (MB) |")
    print("|----------|----------|--------|----------|")
    for k, (acc, _, params, ram) in results.items():
        print(f"| {k:<8} | {acc:.4f}   | {params:<6} | {ram:.2f}     |")
    # Plot confusion matrices
    y_true = np.argmax(y_test, axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (k, (_, preds, _, _)) in enumerate(results.items()):
        cm = compute_confusion_matrix(y_true, preds)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues')
        axes[idx].set_title(f'{k} Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    plt.suptitle(f'{dataset_name} - Confusion Matrices')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Credit Risk
    X_train, X_test, y_train, y_test = load_credit_data()
    compare_all_models_on_dataset(X_train, X_test, y_train, y_test, "Credit Risk", epochs=5)
    # Congressional Voting
    X_train, X_test, y_train, y_test = load_voting_data()
    compare_all_models_on_dataset(X_train, X_test, y_train, y_test, "Congressional Voting", epochs=5)
