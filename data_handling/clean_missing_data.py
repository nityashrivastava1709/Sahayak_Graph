
import pandas as pd

input_file = "data/interim/updated_data_deduplicated.csv"
output_file = "data/interim/updated_data_cleaned.csv"

df = pd.read_csv(input_file)

print("Rows before cleaning:", len(df))
print("Columns before cleaning:", len(df.columns))

# Remove only the column confirmed to be completely empty
df = df.drop(columns=["Unnamed: 9"])

print("\nRows after cleaning:", len(df))
print("Columns after cleaning:", len(df.columns))
print("Remaining columns:", df.columns.tolist())

df.to_csv(output_file, index=False)

print("\nCleaned dataset saved to:", output_file)
