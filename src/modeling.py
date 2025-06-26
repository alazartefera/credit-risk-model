import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os

def load_data():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    X = np.load(os.path.join(base_dir, 'X.npy'))
    y = np.load(os.path.join(base_dir, 'y.npy'))
    return X, y

def evaluate_model(y_true, y_pred, y_proba):
    print("📊 Evaluation Metrics:")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_true, y_proba):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def main():
    X, y = load_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate
    evaluate_model(y_test, y_pred, y_proba)

    # Save model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(model, os.path.join(model_path, 'model.joblib'))
    print(f"✅ Model saved to {os.path.join(model_path, 'model.joblib')}")

if __name__ == "__main__":
    main()
