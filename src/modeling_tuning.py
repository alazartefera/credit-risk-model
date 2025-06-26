import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer
import joblib
import os
from scipy.stats import uniform, randint

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False
    print("⚠️ XGBoost not installed; skipping.")

def load_data():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    X = np.load(os.path.join(base_dir, 'X.npy'))
    y = np.load(os.path.join(base_dir, 'y.npy'))
    return X, y

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42)
    }
    param_distributions = {
        'LogisticRegression': {
            'C': uniform(0.01, 10),
            'penalty': ['l2'],  # 'l1' can be added if solver supports
            'solver': ['lbfgs']
        },
        'RandomForest': {
            'n_estimators': randint(50, 300),
            'max_depth': randint(3, 20),
            'min_samples_split': randint(2, 10)
        }
    }

    if xgb_available:
        models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        param_distributions['XGBoost'] = {
            'n_estimators': randint(50, 300),
            'max_depth': randint(3, 15),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.5, 0.5)
        }

    best_estimators = {}

    for name, model in models.items():
        print(f"🔍 Tuning {name}...")
        search = RandomizedSearchCV(
            model,
            param_distributions=param_distributions[name],
            n_iter=20,
            scoring=make_scorer(roc_auc_score, needs_proba=True),
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        print(f"Best params for {name}: {search.best_params_}")
        best_estimators[name] = search.best_estimator_

        # Evaluate on test set
        y_proba = search.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_proba)
        print(f"Test ROC AUC for {name}: {test_auc:.4f}\n")

    # Save best model (choose by test AUC)
    best_model_name = max(best_estimators, key=lambda k: roc_auc_score(y_test, best_estimators[k].predict_proba(X_test)[:, 1]))
    best_model = best_estimators[best_model_name]

    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'best_model.joblib')
    joblib.dump(best_model, model_path)
    print(f"✅ Saved best model ({best_model_name}) to {model_path}")

if __name__ == "__main__":
    main()
