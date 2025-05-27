import matplotlib.pyplot as plt
import numpy as np

def plot_comparison():
    # Data from your comparison
    configs = ["Credit Risk", "Voting"]
    scratch_acc = [0.7526, 1.0000]
    pytorch_acc = [0.6375, 0.7826]
    
    # Bar chart for accuracy
    x = np.arange(len(configs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7,5))
    rects1 = ax.bar(x - width/2, scratch_acc, width, label='Scratch NN', color='#4C72B0')
    rects2 = ax.bar(x + width/2, pytorch_acc, width, label='PyTorch NN', color='#55A868')
    
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy Comparison: Scratch vs PyTorch NN')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig('results/nn_accuracy_comparison.png')
    plt.show()

if __name__ == "__main__":
    plot_comparison()
