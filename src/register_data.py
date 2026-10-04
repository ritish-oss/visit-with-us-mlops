import sys
import pandas as pd
from pathlib import Path

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "Occupation", "Gender", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "OwnCar",
    "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
    "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups",
    "DurationOfPitch"
]

def register_and_validate(data_path="data/travel_package.csv"):
    file = Path(data_path)
    if not file.exists():
        raise FileNotFoundError(f"Missing dataset at {data_path}")

    df = pd.read_csv(file)
    print(f"Dataset successfully loaded. Total rows: {len(df)}, Columns: {len(df.columns)}")

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Schema Validation Failed! Missing: {missing}")

    print("All 20 expected columns verified.")
    print("--- TARGET SUMMARY ---")
    print(df["ProdTaken"].value_counts(normalize=True))
    print("----------------------")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/travel_package.csv"
    register_and_validate(path)
