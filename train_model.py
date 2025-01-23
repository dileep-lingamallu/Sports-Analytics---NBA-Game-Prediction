from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import pandas as pd

def train_and_evaluate():
    # Load preprocessed datasets
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv').squeeze()  # Ensure it's a Series
    y_test = pd.read_csv('y_test.csv').squeeze()    # Ensure it's a Series

    # Train Random Forest Model
    print("Training Random Forest Model...")
    rf_model = RandomForestClassifier(class_weight="balanced", random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    print("\nRandom Forest Performance:")
    print(classification_report(y_test, y_pred_rf))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))

    # Train XGBoost Model
    print("\nTraining XGBoost Model...")
    xgb_model = XGBClassifier(scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
                              random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    print("\nXGBoost Performance:")
    print(classification_report(y_test, y_pred_xgb))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_xgb))

    # Calculate and display ROC-AUC score
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob_xgb)
    print(f"\nXGBoost ROC-AUC Score: {roc_auc:.2f}")

    # Save the best-performing model
    joblib.dump(xgb_model, 'xgboost_model.pkl')
    print("XGBoost model saved as 'xgboost_model.pkl'.")

if __name__ == "__main__":
    train_and_evaluate()
