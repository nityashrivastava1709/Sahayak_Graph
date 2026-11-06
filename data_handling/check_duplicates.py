
import pandas as pd

file_path = "data/interim/updated_data.csv"

df = pd.read_csv(file_path)

print("Original number of rows:", len(df))

# Check exact duplicates across every column
exact_duplicates = df.duplicated()
print("Exact duplicate rows:", exact_duplicates.sum())

# Normalize scheme names only for comparison; don't change the data
df["_name_check"] = (
    df["scheme_name"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

# Check repeated scheme name + level combinations
name_level_duplicates = df.duplicated(
    subset=["_name_check", "level"],
    keep=False
)

print(
    "Rows involved in repeated scheme name + level combinations:",
    name_level_duplicates.sum()
)

print("\nExamples of repeated scheme name + level combinations:")
print(
    df.loc[
        name_level_duplicates,
        ["_name_check", "scheme_name", "level", "slug"]
    ]
    .sort_values(["_name_check", "level"])
    .head(30)
    .to_string(index=False)
)

# Check names that become identical after lowercase + trimming
name_duplicates = df.duplicated(
    subset=["_name_check"],
    keep=False
)

print(
    "\nRows involved in repeated names (ignoring case and outer spaces):",
    name_duplicates.sum()
)

print("\nExact duplicate rows across all columns (first 10):")
print(df.loc[exact_duplicates].drop(columns="_name_check").head(10).to_string(index=False))

print("\nDetailed comparison of repeated schemes:")

repeated_schemes = df[
    df["_name_check"].isin([
        "establishment of goat unit (10 +1)",
        "valmiki chhatravritti yojana"
    ])
]

for scheme_name, group in repeated_schemes.groupby("_name_check"):
    print(f"\nScheme: {scheme_name}")
    print("Number of records:", len(group))

    # Count distinct values in each column
    for column in df.columns.drop("_name_check"):
        distinct_count = group[column].nunique(dropna=False)

        if distinct_count > 1:
            print(f"  {column}: {distinct_count} different values")