import pandas as pd

input_file = "data/interim/updated_data_text_cleaned.csv"
output_file = "data/interim/updated_data_category_cleaned.csv"

df = pd.read_csv(input_file)

mapping = {
    "Socialwelfare & Empowerment": "Social welfare & Empowerment"
}

def normalize(value):
    if pd.isna(value):
        return value

    value = " ".join(value.split())
    return value.replace(
        "Socialwelfare & Empowerment",
        "Social welfare & Empowerment"
    )

df["schemeCategory"] = df["schemeCategory"].apply(normalize)

df.to_csv(output_file, index=False, encoding="utf-8-sig")

# Save documented mapping
pd.DataFrame(
    mapping.items(),
    columns=["original_category", "canonical_category"]
).to_csv("data/interim/category_mapping.csv", index=False)

print("Category normalization completed!")
print("Rows:", len(df))
print("Unique category combinations:",
      df["schemeCategory"].nunique())
print("Output:", output_file)