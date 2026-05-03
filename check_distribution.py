import pandas as pd
import re

df = pd.read_csv("dataset.csv")

# remove HTML
def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = text.lower()
    return text

df["clean_text"] = df["AbstractText"].apply(clean_text)

# remove weak journals
counts = df["JournalName"].value_counts()
valid_journals = counts[counts >= 10].index

df = df[df["JournalName"].isin(valid_journals)]

print("After cleaning:")
print("Rows:", df.shape)
print("Journals:", df["JournalName"].nunique())