import numpy as np

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

def count_parameters(model):
    total = 0
    for layer in getattr(model, 'layers', []):
        if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
            total += np.prod(layer.weights.shape) + np.prod(layer.biases.shape)
    return total

def estimate_ram(model):
    param_count = count_parameters(model)
    return (param_count * 4) / (1024 ** 2)  # MB
