# NN Visualization Script
"""
Load saved results from nn_comparison.py and generate visualizations without retraining.
"""
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from nn_utils import compute_confusion_matrix

def load_results(results_path):
    with open(results_path, 'rb') as f:
        return pickle.load(f)

def plot_training_curves(histories, dataset_name):
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
    plt.savefig(f"results/{dataset_name.replace(' ', '_').lower()}_training_curves.png")
    plt.show()

def plot_confusion_matrices(results, y_test, dataset_name):
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
    plt.savefig(f"results/{dataset_name.replace(' ', '_').lower()}_confusion_matrices.png")
    plt.show()

def plot_bar_charts(results, dataset_name):
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
    plt.savefig(f"results/{dataset_name.replace(' ', '_').lower()}_bar_charts.png")
    plt.show()

def print_table_summary(results):
    values = {
        'Custom': [results['Custom'][0], results['Custom'][2], results['Custom'][3]],
        'LLM': [results['LLM'][0], results['LLM'][2], results['LLM'][3]],
        'PyTorch': [results['PyTorch'][0], results['PyTorch'][2], results['PyTorch'][3]]
    }
    df = pd.DataFrame({
        'Model': list(values.keys()),
        'Accuracy': [v[0] for v in values.values()],
        'Params': [v[1] for v in values.values()],
        'RAM (MB)': [v[2] for v in values.values()]
    })
    print('\nTabular Summary:')
    print(df.to_string(index=False))

def main():
    results_dir = "results"
    for fname in os.listdir(results_dir):
        if fname.endswith("_results.pkl"):
            data = load_results(os.path.join(results_dir, fname))
            print(f"\n=== Visualizing: {data['dataset_name']} ===")
            print_table_summary(data['results'])
            plot_bar_charts(data['results'], data['dataset_name'])
            plot_confusion_matrices(data['results'], data['y_test'], data['dataset_name'])
            plot_training_curves(data['histories'], data['dataset_name'])

if __name__ == "__main__":
    main()
