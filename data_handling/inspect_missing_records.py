import pandas as pd

file_path = "data/interim/updated_data_cleaned.csv"
df = pd.read_csv(file_path)

for column in ["application", "documents", "tags"]:
    missing = df[column].isna() | (
        df[column].astype("string").str.strip().eq("").fillna(False)
    )

    print(f"\n--- {column} ---")
    print("Missing or blank records:", missing.sum())

    print(
        df.loc[missing, ["scheme_name", column]]
        .to_string(index=False)
    )

print("\nTotal rows preserved:", len(df))