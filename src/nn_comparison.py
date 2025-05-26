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
import pandas as pd
import pickle
import os

# Helper to run and evaluate a model

def run_and_evaluate(model_class, X_train, y_train, X_test, y_test, model_args=None, train_kwargs=None, epochs=5):
    if model_args is None:
        model_args = {}
    if train_kwargs is None:
        train_kwargs = {}
    model = model_class(**model_args) if model_class is not None else None
    history = None
    if hasattr(model, 'add_layer'):
        model.add_layer(input_size=X_train.shape[1], output_size=32, activation='leaky_relu')
        model.add_dropout_layer(rate=0.1)
        model.add_layer(input_size=32, output_size=2, activation=None)
        # --- Collect history if available ---
        history = model.train(X_train, y_train, epochs=epochs, learning_rate=0.01, batch_size=32, return_history=True)
        preds = model.predict(X_test)
        acc = model.evaluate(X_test, y_test)
        params = count_parameters(model)
        ram = estimate_ram(model)
        return acc, preds, model, params, ram, history
    else:
        # PyTorch
        acc, preds, model, history = train_torch_nn(
            X_train, y_train, X_test, y_test,
            hidden_sizes=[32], activation='leaky_relu', dropout=0.1, epochs=epochs, lr=0.01, return_history=True
        )
        params = sum(p.numel() for p in model.parameters())
        ram = params * 4 / (1024**2)
        return acc, preds, model, params, ram, history

# Main comparison for both datasets

def compare_all_models_on_dataset(X_train, X_test, y_train, y_test, dataset_name, epochs=5, save_results=True, results_dir="results"):
    print(f"\n===== {dataset_name} =====")
    results = {}
    histories = {}
    # Custom
    acc_c, preds_c, model_c, params_c, ram_c, hist_c = run_and_evaluate(CustomNN, X_train, y_train, X_test, y_test, epochs=epochs)
    results['Custom'] = (acc_c, preds_c, params_c, ram_c)
    histories['Custom'] = hist_c
    # LLM
    acc_l, preds_l, model_l, params_l, ram_l, hist_l = run_and_evaluate(LLMNN, X_train, y_train, X_test, y_test, epochs=epochs)
    results['LLM'] = (acc_l, preds_l, params_l, ram_l)
    histories['LLM'] = hist_l
    # PyTorch
    acc_p, preds_p, model_p, params_p, ram_p, hist_p = run_and_evaluate(None, X_train, y_train, X_test, y_test, epochs=epochs)
    results['PyTorch'] = (acc_p, preds_p, params_p, ram_p)
    histories['PyTorch'] = hist_p
    # Save all results to disk
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f"{dataset_name.replace(' ', '_').lower()}_results.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump({
                'results': results,
                'histories': histories,
                'y_test': y_test,
                'dataset_name': dataset_name
            }, f)
        print(f"[Saved results to {save_path}]")
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
    # --- Visualization: Bar chart for accuracy, params, RAM ---
    metrics = ['Accuracy', 'Params', 'RAM (MB)']
    values = {
        'Custom': [results['Custom'][0], results['Custom'][2], results['Custom'][3]],
        'LLM': [results['LLM'][0], results['LLM'][2], results['LLM'][3]],
        'PyTorch': [results['PyTorch'][0], results['PyTorch'][2], results['PyTorch'][3]]
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, metric in enumerate(metrics):
        axes[i].bar(values.keys(), [v[i] for v in values.values()], color=['skyblue', 'orange', 'green'])
        axes[i].set_title(metric)
        axes[i].set_ylabel(metric)
        if metric == 'Accuracy':
            axes[i].set_ylim(0, 1)
    plt.suptitle(f'{dataset_name} - Model Comparison Metrics')
    plt.tight_layout()
    plt.show()
    # --- Training Curves ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for model_name, hist in histories.items():
        if hist is not None:
            axes[0].plot(hist['loss'], label=model_name)
            axes[1].plot(hist['accuracy'], label=model_name)
    axes[0].set_title('Training Loss per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].set_title('Training Accuracy per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.suptitle(f'{dataset_name} - Training Curves')
    plt.tight_layout()
    plt.show()
    # --- Table summary (printed) ---
    df = pd.DataFrame({
        'Model': list(values.keys()),
        'Accuracy': [v[0] for v in values.values()],
        'Params': [v[1] for v in values.values()],
        'RAM (MB)': [v[2] for v in values.values()]
    })
    print('\nTabular Summary:')
    print(df.to_string(index=False))

if __name__ == "__main__":
    # Credit Risk
    X_train, X_test, y_train, y_test = load_credit_data()
    compare_all_models_on_dataset(X_train, X_test, y_train, y_test, "Credit Risk", epochs=5)
    # Congressional Voting
    X_train, X_test, y_train, y_test = load_voting_data()
    compare_all_models_on_dataset(X_train, X_test, y_train, y_test, "Congressional Voting", epochs=5)
