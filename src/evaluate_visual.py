import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import joblib
import os

def load_data():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    X = np.load(os.path.join(base_dir, 'X.npy'))
    y = np.load(os.path.join(base_dir, 'y.npy'))
    return X, y

def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

def main():
    X, y = load_data()

    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.joblib')
    model = joblib.load(model_path)

    y_proba = model.predict_proba(X)[:,1]
    y_pred = model.predict(X)

    plot_roc_curve(y, y_proba)
    plot_confusion_matrix(y, y_pred)

if __name__ == "__main__":
    main()
