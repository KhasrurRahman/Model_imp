# Grid Search Visualization Script
"""
Parse and visualize grid search results for model hyperparameters.
Generates heatmaps and bar charts for best accuracy by configuration.
"""
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def parse_grid_search_file(filepath):
    results = []
    best_config = None
    best_acc = -1
    with open(filepath, 'r') as f:
        for line in f:
            m = re.match(r"Activation: (\w+), Hidden Layers: (\d+), Nodes per Layer: (\d+) => Accuracy: ([0-9.]+)", line)
            if m:
                act, layers, nodes, acc = m.groups()
                results.append({
                    'Activation': act,
                    'Hidden Layers': int(layers),
                    'Nodes per Layer': int(nodes),
                    'Accuracy': float(acc)
                })
            if line.startswith('Best Configuration:'):
                best_config = f.readline().strip()
    df = pd.DataFrame(results)
    return df, best_config

def plot_all_heatmaps(df, dataset_name):
    activations = df['Activation'].unique()
    n = len(activations)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    for i, act in enumerate(activations):
        pivot = df[df['Activation']==act].pivot(index='Hidden Layers', columns='Nodes per Layer', values='Accuracy')
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', ax=axes[i])
        axes[i].set_title(f'Activation: {act}')
        axes[i].set_ylabel('Hidden Layers')
        axes[i].set_xlabel('Nodes per Layer')
    plt.suptitle(f'{dataset_name} - Accuracy Heatmaps by Activation', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"results/{dataset_name.lower().replace(' ', '_')}_grid_heatmaps_all.png")
    plt.show()

def plot_best_bar(df, dataset_name):
    # Bar plot: best accuracy for each activation
    bests = df.groupby('Activation')['Accuracy'].max().reset_index()
    plt.figure(figsize=(6,4))
    sns.barplot(data=bests, x='Activation', y='Accuracy', palette='Set2')
    plt.title(f'{dataset_name} - Best Accuracy by Activation')
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(f"results/{dataset_name.lower().replace(' ', '_')}_grid_best_bar.png")
    plt.show()

def main():
    files = [
        ('Credit Risk', 'results/grid_search_results_credit_risk.txt'),
        ('Congressional Voting', 'results/grid_search_results_congressional_voting.txt')
    ]
    for dataset_name, filepath in files:
        if os.path.exists(filepath):
            df, best_config = parse_grid_search_file(filepath)
            print(f"\n=== {dataset_name} Grid Search Results ===")
            print(df)
            print(f"Best Configuration: {best_config}")
            plot_all_heatmaps(df, dataset_name)
            plot_best_bar(df, dataset_name)
        else:
            print(f"File not found: {filepath}")

if __name__ == "__main__":
    main()
