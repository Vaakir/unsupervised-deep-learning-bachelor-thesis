import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

def plot_confusion_matrix(y_pred, y_test):
    y_true_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true_labels, y_pred_labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    # Compute accuracy
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    
    # Check if binary classification (2 classes)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        recall = recall_score(y_true_labels, y_pred_labels)

        plt.figtext(0.5, -0.1, f"Accuracy: {accuracy:.4f}\nSpecificity: {specificity:.4f}\nRecall: {recall:.4f}", 
                    ha="center", fontsize=12)
    else:
        plt.figtext(0.5, -0.1, f"Accuracy: {accuracy:.4f}", ha="center", fontsize=12)
