import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/flights.csv")
PROCESSED_PATH = Path("data/processed/flights_sample.csv")

SAMPLE_SIZE = 200_000  # safe for local ML

def main():
    print("Reading raw dataset...")
    df = pd.read_csv(RAW_PATH)

    print(f"Original rows: {len(df)}")

    df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)

    print(f"Sampled rows: {len(df_sample)}")

    df_sample.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved sampled data to {PROCESSED_PATH}")

if __name__ == "__main__":
    main()
