import os
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def train_model():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    X_train = train_df.drop(columns=["ProdTaken"])
    y_train = train_df["ProdTaken"]
    X_test = test_df.drop(columns=["ProdTaken"])
    y_test = test_df["ProdTaken"]

    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImTake a breath. Two hours is enough time to get a clean repository set up, execute the workflow, deploy the app, and take the screenshots your rubric requires. 

Follow this exact **pin-to-pin, emergency checklist**.

---

### Step 1: Set Up Your Project Folder on Your Computer (10 mins)

1. Open your terminal (or Command Prompt / VS Code).
2. Create and move into your project folder:
   ```bash
   mkdir visit-with-us
   cd visit-with-us
   mkdir -p data src .github/workflows models
