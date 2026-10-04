import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    df = pd.read_csv("data/travel_package.csv")

    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["ProdTaken"]
    )

    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    print("Splits created: data/train.csv and data/test.csv")

if __name__ == "__main__":
    prepare_data()
