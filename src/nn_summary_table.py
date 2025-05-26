# NN Summary Table Script
"""
Loads all results and prints a single summary table comparing all models and datasets.
"""
import os
import pickle
import pandas as pd

def load_results(results_path):
    with open(results_path, 'rb') as f:
        return pickle.load(f)

def main():
    results_dir = "results"
    summary_rows = []
    for fname in os.listdir(results_dir):
        if fname.endswith("_results.pkl"):
            data = load_results(os.path.join(results_dir, fname))
            dataset = data['dataset_name']
            for model, (acc, _, params, ram) in data['results'].items():
                row = {
                    'Dataset': dataset,
                    'Model': model,
                    'Accuracy': acc,
                    'Params': params,
                    'RAM (MB)': ram
                }
                summary_rows.append(row)
    df = pd.DataFrame(summary_rows)
    print("\n=== Neural Network Comparison Summary Table ===")
    print(df.to_string(index=False))
    # Optionally, save as CSV for further use
    df.to_csv(os.path.join(results_dir, "nn_summary_table.csv"), index=False)
    print(f"\n[Saved summary table to {os.path.join(results_dir, 'nn_summary_table.csv')}]\n")

if __name__ == "__main__":
    main()
