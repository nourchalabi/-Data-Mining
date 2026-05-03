import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# Load dataset
df = pd.read_csv("dataset.csv")


# Clean text
def clean_text(text):
    text = re.sub(r"<.*?>", "", str(text))
    text = text.lower()
    return text


df["clean_text"] = df["AbstractText"].apply(clean_text)


# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = vectorizer.fit_transform(df["clean_text"])


# Number of topic clusters
k = 10

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X)


# Show top words for each cluster
terms = vectorizer.get_feature_names_out()

print("\nTopic clusters:\n")

for cluster_id in range(k):
    center = kmeans.cluster_centers_[cluster_id]
    top_indices = center.argsort()[-10:][::-1]
    top_words = [terms[i] for i in top_indices]

    print(f"Cluster {cluster_id}:")
    print(", ".join(top_words))
    print()


# Save result
df[["AbstractText", "JournalName", "Cluster"]].to_csv(
    "clustered_dataset.csv",
    index=False
)

print("Clustered dataset saved as clustered_dataset.csv")