import pandas as pd

df = pd.read_csv("data/interim/updated_data_text_cleaned.csv")

print(df["schemeCategory"].value_counts(dropna=False).to_string())