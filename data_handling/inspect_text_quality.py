
import pandas as pd
import re

file_path = "data/interim/updated_data_cleaned.csv"
df = pd.read_csv(file_path)

text_columns = [
    "scheme_name",
    "details",
    "benefits",
    "eligibility",
    "application",
    "documents",
    "tags",
]

print("Dataset rows:", len(df))

for column in text_columns:
    text = df[column].dropna().astype(str)

    leading_trailing = text[
        text.str.strip().ne(text)
    ].shape[0]

    repeated_spaces = text[
        text.str.contains(r" {2,}", regex=True, na=False)
    ].shape[0]

    tabs_or_newlines = text[
        text.str.contains(r"[\t\r\n]", regex=True, na=False)
    ].shape[0]

    print(f"\n--- {column} ---")
    print("Values with leading/trailing whitespace:", leading_trailing)
    print("Values with repeated spaces:", repeated_spaces)
    print("Values with tabs or newlines:", tabs_or_newlines)

print("\nSample descriptions for manual review:")
for _, row in df.head(5).iterrows():
    print(f"\nScheme: {row['scheme_name']}")
    print(f"Details: {str(row['details'])[:400]}")


print("\nExact locations of repeated spaces:")

for column in ["scheme_name", "details", "benefits", "eligibility",
               "application", "documents", "tags"]:
    count = 0

    for idx, value in df[column].items():
        if isinstance(value, str):
            matches = list(re.finditer(r" {2,}", value))

            for match in matches:
                start = max(0, match.start() - 40)
                end = min(len(value), match.end() + 40)

                print(f"\nColumn: {column}, row index: {idx}")
                print(repr(value[start:end]))

                count += 1
                if count == 3:
                    break

        if count == 3:
            break

print("\nInspecting long spaces in details column:")

count = 0

for idx, value in df["details"].items():
    if isinstance(value, str):
        matches = re.finditer(r" {5,}", value)

        for match in matches:
            start = max(0, match.start() - 80)
            end = min(len(value), match.end() + 80)

            print(f"\nRow index: {idx}")
            print("Context:", repr(value[start:end]))
            print("Number of consecutive spaces:", len(match.group()))

            count += 1

            if count == 10:
                break

    if count == 10:
        break

if count == 0:
    print("No long spaces found.")


print("\nInspecting possible missing spaces between words:")

pattern = re.compile(r"[a-z][A-Z]")

for column in ["scheme_name", "details", "benefits",
               "eligibility", "application", "documents", "tags"]:

    count = 0

    for idx, value in df[column].items():
        if isinstance(value, str):
            matches = list(pattern.finditer(value))

            for match in matches[:3]:
                start = max(0, match.start() - 40)
                end = min(len(value), match.end() + 40)

                print(f"\nColumn: {column}, Row: {idx}")
                print(repr(value[start:end]))

                count += 1

            if count >= 3:
                break