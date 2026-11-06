
import pandas as pd

file_path = "data/interim/updated_data_deduplicated.csv"

df = pd.read_csv(file_path)

print("Total rows:", len(df))
print("\nMissing values in each column:")
print(df.isna().sum())

print("\nRows with missing values:")
missing_rows = df[df.isna().any(axis=1)]
print("Number of rows:", len(missing_rows))

print("\nRows missing application details:")
print(df[df["application"].isna()][
    ["scheme_name", "level", "application"]
].to_string(index=False))

print("\nRows missing document details:")
print(df[df["documents"].isna()][
    ["scheme_name", "level", "documents"]
].to_string(index=False))

print("\nRows missing tags:")
print(df[df["tags"].isna()][
    ["scheme_name", "level", "tags"]
].to_string(index=False))


print("\nChecking the completely empty column:")

column = "Unnamed: 9"

print("Non-missing values:", df[column].notna().sum())
print("Unique non-missing values:", df[column].dropna().unique())

print("\nChecking text fields for blank strings:")

for column in ["application", "documents", "tags"]:
    blank_count = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"{column}: {blank_count} blank or missing values")
