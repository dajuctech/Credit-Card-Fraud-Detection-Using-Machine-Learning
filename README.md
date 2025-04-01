# Credit Card Fraud Detection

This project uses machine learning to spot fraudulent credit card transactions. It walks you through data cleanup, exploring the data, feature engineering, training several models, and comparing their performance.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Model Comparison](#model-comparison)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

---

## Project Overview

This project builds and compares different machine learning models to decide if a credit card transaction is fraud. We start from raw data, clean and analyze it, then build models like Logistic Regression, Random Forest, KNN, Naive Bayes, XGBoost, and SVM.

---

## Data Collection

- **Source:** The data is from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data).
- **Details:** The dataset has 31 columns (28 anonymized features, the transaction amount, and a target column `Class` where 0 is normal and 1 is fraud).

---

## Data Preprocessing

We use a custom class to:
- Check for missing values and duplicates.
- Remove duplicates (1081 records were dropped).
- Look at summary statistics, data types, and outliers.
- Visualize data distributions and correlations.

---

## Exploratory Data Analysis

We explore the dataset to understand:
- The imbalance between normal and fraud transactions.
- Distribution of transaction amounts and time.
- Correlations between features.
- Outliers and overall feature behavior using histograms, box plots, and heatmaps.

---

## Feature Engineering

Key steps include:
- Dropping the `Time` column (it doesn’t help the prediction).
- Splitting data into features (`X`) and target (`y`).
- Normalizing the `Amount` feature.
- Handling class imbalance using techniques like oversampling, undersampling, and SMOTE.

---

## Modeling

We trained several models using different versions of the data:

### Logistic Regression
- Good accuracy on the original data.
- Better fraud detection when using balanced datasets.

### Random Forest
- Performs very well on balanced data.
- Slightly lower recall on the imbalanced dataset.

### K-Nearest Neighbors (KNN)
- Works best on oversampled and SMOTE data.
- Imbalanced data slightly reduces its fraud detection ability.

### Naive Bayes
- Benefits significantly from balancing the data.
- The original imbalanced data performs poorly.

### XGBoost
- Achieves near-perfect accuracy on all data versions.
- Further checks are needed to avoid overfitting.

### Support Vector Machine (SVM)
- Performs well when the data is scaled.
- Undersampling gives a balanced result, though with a small drop in overall accuracy.

---

## Model Comparison

A comparison class gathers accuracy scores from all models across different data treatments. A heatmap shows that using techniques like oversampling and SMOTE boosts performance. Overall, XGBoost and Random Forest lead the pack.

---

## Conclusion

Proper data handling (scaling, oversampling, SMOTE) improves model performance significantly. While simple models like Logistic Regression and Naive Bayes benefit from balanced data, ensemble methods like Random Forest and XGBoost deliver the best results for detecting credit card fraud.

---

## How to Run

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection

2. **Install Dependencies:**
```bash
pip install -r requirements.txt

3. **Run the Notebook:**
Open `CreditCardFraudDetection.ipynb` in Jupyter Notebook or Google Colab.

# Dependencies

* Python 3.x
* pandas
* numpy
* seaborn
* matplotlib
* scikit-learn
* imbalanced-learn
* xgboost


