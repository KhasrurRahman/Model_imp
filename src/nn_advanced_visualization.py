# Advanced NN Visualization Script
"""
Additional visualizations: ROC/AUC, precision-recall, per-class accuracy, classification report.
Saves all plots as PNGs for use in slides.
"""
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report, confusion_matrix
from nn_utils import compute_confusion_matrix

def load_results(results_path):
    with open(results_path, 'rb') as f:
        return pickle.load(f)

def plot_roc_curves(results, y_test, dataset_name):
    y_true = np.argmax(y_test, axis=1)
    plt.figure(figsize=(7, 5))
    for model_name, (_, preds, _, _) in results.items():
        # If preds are class indices, make one-hot
        if preds.ndim == 1 or preds.shape[1] == 1:
            y_score = np.eye(np.max(y_true)+1)[preds]
        else:
            y_score = preds
        fpr, tpr, _ = roc_curve(y_true, y_score[:,1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_name} - ROC Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{dataset_name.replace(' ', '_').lower()}_roc_curve.png")
    plt.close()

def plot_pr_curves(results, y_test, dataset_name):
    y_true = np.argmax(y_test, axis=1)
    plt.figure(figsize=(7, 5))
    for model_name, (_, preds, _, _) in results.items():
        if preds.ndim == 1 or preds.shape[1] == 1:
            y_score = np.eye(np.max(y_true)+1)[preds]
        else:
            y_score = preds
        precision, recall, _ = precision_recall_curve(y_true, y_score[:,1])
        plt.plot(recall, precision, label=model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{dataset_name} - Precision-Recall Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{dataset_name.replace(' ', '_').lower()}_pr_curve.png")
    plt.close()

def plot_per_class_accuracy(results, y_test, dataset_name):
    y_true = np.argmax(y_test, axis=1)
    accs = {}
    for model_name, (_, preds, _, _) in results.items():
        per_class = []
        for c in np.unique(y_true):
            per_class.append((preds == c).sum() / (y_true == c).sum())
        accs[model_name] = per_class
    df = pd.DataFrame(accs, index=[f'Class {c}' for c in np.unique(y_true)])
    df.plot(kind='bar', figsize=(8,5))
    plt.title(f'{dataset_name} - Per-Class Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(f"results/{dataset_name.replace(' ', '_').lower()}_per_class_accuracy.png")
    plt.close()

def print_classification_reports(results, y_test, dataset_name):
    y_true = np.argmax(y_test, axis=1)
    for model_name, (_, preds, _, _) in results.items():
        print(f"\nClassification Report for {model_name} on {dataset_name}:")
        print(classification_report(y_true, preds))

def main():
    results_dir = "results"
    for fname in os.listdir(results_dir):
        if fname.endswith("_results.pkl"):
            data = load_results(os.path.join(results_dir, fname))
            print(f"\n=== Advanced Visualizations: {data['dataset_name']} ===")
            plot_roc_curves(data['results'], data['y_test'], data['dataset_name'])
            plot_pr_curves(data['results'], data['y_test'], data['dataset_name'])
            plot_per_class_accuracy(data['results'], data['y_test'], data['dataset_name'])
            print_classification_reports(data['results'], data['y_test'], data['dataset_name'])

if __name__ == "__main__":
    main()
