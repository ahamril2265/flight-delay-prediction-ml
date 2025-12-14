import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/flights_sample.csv")
OUTPUT_PATH = Path("data/processed/flights_labeled.csv")

def main():
    df = pd.read_csv(INPUT_PATH)
    print("Initial rows:", len(df))

    # Remove cancelled flights (important)
    if "CANCELLED" in df.columns:
        df = df[df["CANCELLED"] == 0]
        print("After removing cancelled flights:", len(df))

    # Create target variable
    df["IS_DELAYED"] = (df["DEP_DELAY"] > 15).astype(int)

    # Drop leakage columns
    df = df.drop(columns=[
        "DEP_DELAY",
        "CANCELLED"
    ], errors="ignore")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved labeled dataset to {OUTPUT_PATH}")
    print(df["IS_DELAYED"].value_counts(normalize=True))

if __name__ == "__main__":
    main()
