import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_history(train_loss, val_loss, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='Val')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_cam(sequence, scores, save_path):
    # Normalize
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    plt.figure(figsize=(15, 3))
    plt.bar(range(len(sequence)), scores, color='red', alpha=0.6)
    plt.title(f"CAM Feature Importance (Seq Len: {len(sequence)})")
    plt.savefig(save_path)
    plt.close()