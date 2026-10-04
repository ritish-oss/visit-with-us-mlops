import os
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def train():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    X_train = train_df.drop(columns=["ProdTaken"])
    y_train = train_df["ProdTaken"]
    X_test = test_df.drop(columns=["ProdTaken"])
    y_test = test_df["ProdTaken"]

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols)
        ]
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", XGBClassifier(random_state=42, eval_metric="logloss"))
    ])

    params = {
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [3, 5],
        "clf__learning_rate": [0.05, 0.1]
    }

    grid = GridSearchCV(pipe, params, cv=3, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_pipeline = grid.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "best_params": grid.best_params_,
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, "models/model.joblib")
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Model saved. Test F1: {metrics['f1_score']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    train()
