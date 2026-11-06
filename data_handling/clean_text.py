
import pandas as pd
import re
from pathlib import Path

# --------------------------------------------------
# 1. Define input and output files
# --------------------------------------------------

input_file = Path("data/interim/updated_data_cleaned.csv")
output_file = Path("data/interim/updated_data_text_cleaned.csv")

# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

df = pd.read_csv(input_file)

original_rows = len(df)
original_columns = list(df.columns)
original_shape = df.shape

print("Original dataset shape:", original_shape)

# --------------------------------------------------
# 3. Clean repeated spaces in text columns
# --------------------------------------------------

# Normalize runs of 2-4 ordinary spaces.
# Preserve runs of 5 or more spaces because they may
# represent table formatting.

pattern = re.compile(r"(?<! ) {2,4}(?! )")

total_changes = 0

for column in df.select_dtypes(include=["object"]).columns:

    def clean_text(value):
        global total_changes

        if isinstance(value, str):
            cleaned_value, changes = pattern.subn(" ", value)
            total_changes += changes
            return cleaned_value

        return value

    df[column] = df[column].apply(clean_text)

# --------------------------------------------------
# 4. Verify that no rows or columns were removed
# --------------------------------------------------

assert len(df) == original_rows, "Row count changed!"
assert list(df.columns) == original_columns, "Columns changed!"
assert df.shape == original_shape, "Dataset shape changed!"

# --------------------------------------------------
# 5. Save the cleaned dataset separately
# --------------------------------------------------

df.to_csv(output_file, index=False, encoding="utf-8-sig")

# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("\nText cleaning completed!")
print("Original shape:", original_shape)
print("Final shape:", df.shape)
print("Repeated-space replacements:", total_changes)
print("Rows preserved:", len(df))
print("Columns preserved:", len(df.columns))
print("Output saved to:", output_file)
