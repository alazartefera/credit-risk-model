import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False
    print("⚠️ XGBoost is not installed. Skipping XGBoost model.")

def load_data():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    X = np.load(os.path.join(base_dir, 'X.npy'))
    y = np.load(os.path.join(base_dir, 'y.npy'))
    return X, y

def evaluate_model(y_true, y_pred, y_proba, model_name):
    print(f"\n📊 Evaluation for {model_name}:")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_true, y_proba):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, y_pred, y_proba, model_name)
    return model

def main():
    X, y = load_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }

    if xgb_available:
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1, random_state=42)

    trained_models = {}

    for name, model in models.items():
        trained_models[name] = train_and_evaluate(model, name, X_train, X_test, y_train, y_test)

    # Save best model (for now, Logistic Regression as default)
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(trained_models["LogisticRegression"], os.path.join(model_dir, 'model.joblib'))
    print(f"\n✅ LogisticRegression model saved to {os.path.join(model_dir, 'model.joblib')}")

if __name__ == "__main__":
    main()
