
import pandas as pd

input_file = "data/interim/updated_data.csv"
output_file = "data/interim/updated_data_deduplicated.csv"

df = pd.read_csv(input_file)

print("Rows before cleaning:", len(df))

df_clean = df.drop_duplicates(keep="first")

print("Rows after cleaning:", len(df_clean))
print("Rows removed:", len(df) - len(df_clean))

df_clean.to_csv(output_file, index=False)

print("Cleaned dataset saved to:", output_file)
