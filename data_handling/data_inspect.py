
import pandas as pd

file_path = "data/interim/updated_data.csv"

df = pd.read_csv(file_path)

print("Dataset shape (rows, columns):", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)
