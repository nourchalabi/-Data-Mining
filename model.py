import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import top_k_accuracy_score


# Load dataset
df = pd.read_csv("dataset.csv")


# Clean text
def clean_text(text):
    text = re.sub(r"<.*?>", "", str(text))
    text = text.lower()
    return text


df["clean_text"] = df["AbstractText"].apply(clean_text)


# Remove journals with fewer than 10 articles
counts = df["JournalName"].value_counts()
valid_journals = counts[counts >= 10].index
df = df[df["JournalName"].isin(valid_journals)]


# Shuffle data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# =========================
# 📊 Journal Distribution Chart
# =========================
top_counts = df["JournalName"].value_counts().head(10)

plt.figure(figsize=(10, 5))
top_counts.plot(kind="bar")
plt.title("Top 10 Journals by Number of Articles")
plt.xlabel("Journal")
plt.ylabel("Number of Articles")
plt.xticks(rotation=60, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig("journal_distribution.png")
plt.show()


# =========================
# 🔠 TF-IDF
# =========================
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["JournalName"]


# =========================
# 🔀 Train/Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# 🤖 Train Model
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# =========================
# 📈 Accuracy
# =========================
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)


# =========================
# 📈 Top-5 Accuracy
# =========================
probs = model.predict_proba(X_test)

top5_acc = top_k_accuracy_score(
    y_test,
    probs,
    k=5,
    labels=model.classes_
)

print("Top-5 Accuracy:", top5_acc)


# =========================
# 📊 Accuracy Chart (WITH VALUES)
# =========================
labels = ["Top-1 Accuracy", "Top-5 Accuracy"]
values = [accuracy, top5_acc]

plt.figure(figsize=(6, 4))
bars = plt.bar(labels, values)

plt.title("Model Performance Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

# 🔥 Add values on top of bars
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

plt.tight_layout()
plt.savefig("accuracy_chart.png")
plt.show()


# =========================
# 🎯 Top-5 Prediction Function
# =========================
def predict_top5(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])

    probs = model.predict_proba(vec)[0]
    top5_idx = np.argsort(probs)[-5:][::-1]

    top5_journals = model.classes_[top5_idx]
    top5_probs = probs[top5_idx]

    return list(zip(top5_journals, top5_probs))


# =========================
# 🧪 Test Example
# =========================
sample = df["AbstractText"].iloc[0]

print("\nTop 5 predictions:")
for journal, prob in predict_top5(sample):
    print(f"{journal}: {prob:.4f}")