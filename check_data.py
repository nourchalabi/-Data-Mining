
import pandas as pd


df = pd.read_csv("dataset.csv")

print("Rows and columns:", df.shape)
print("Unique journals:", df["JournalName"].nunique())
print(df.head())