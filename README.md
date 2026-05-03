# 📘 Journal Recommendation System using Data Mining

## 📌 Overview

This project implements a journal recommendation system for computer science research articles.
Given an article abstract, the system predicts the most relevant journals using machine learning techniques.

The system also performs topic clustering to identify major research areas within the dataset.

---

## 🎯 Objectives

* Recommend suitable journals based on article abstracts
* Provide Top-5 journal predictions instead of a single result
* Identify research topics using clustering techniques

---

## 📂 Dataset

* Total records: 23,061
* Total journals: 402 (after filtering)
* Data source: academic publication database
* Features used:

  * AbstractText (input)
  * JournalName (label)

Journals with fewer than 10 samples were removed to improve model performance.

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Removed HTML tags from abstracts
* Converted text to lowercase
* Cleaned and normalized text data

---

### 2. Feature Extraction

* Used **TF-IDF (Term Frequency–Inverse Document Frequency)**
* Converted text into numerical feature vectors

---

### 3. Classification Model

* Model: Logistic Regression
* Train/Test Split: 80% / 20%
* Multi-class classification with 402 journal classes

---

### 4. Top-5 Recommendation

Instead of predicting a single journal, the system returns the top 5 most probable journals using prediction probabilities.

---

### 5. Clustering

* Algorithm: K-Means
* Number of clusters: 10
* Purpose: identify major research topics

---

## 📊 Results

* Top-1 Accuracy: **27.55%**
* Top-5 Accuracy: **54.24%**

The higher Top-5 accuracy shows that the system effectively recommends relevant journals even when the top prediction is not correct.

---

## 🧠 Insights

* Journal recommendation is a difficult multi-class problem due to overlapping research domains
* Providing multiple recommendations is more practical than a single prediction
* Clustering reveals meaningful research topics such as:

  * Data Mining
  * Software Engineering
  * Image Processing
  * Network Systems
  * Security

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install pandas scikit-learn matplotlib
```

---

### 2. Run classification model

```bash
python model.py
```

---

### 3. Run clustering

```bash
python clustering.py
```

---

## 📁 Project Structure

```text
data mining/
│
├── dataset.csv
├── model.py
├── clustering.py
├── journal_distribution.png
├── accuracy_chart.png
├── clustered_dataset.csv
└── README.md
```

---

## 🔧 Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib

---

## 📌 Future Improvements

* Use deep learning models (e.g., BERT)
* Incorporate additional features (keywords, authors)
* Improve classification accuracy
* Build a user interface for real-time recommendations

---

## 👨‍💻 Author

Your Name

---

## 📄 License

This project is for academic purposes.
