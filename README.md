# Credit Card Fraud Detection Project

## Overview

Hey there! I'm Oliver, and welcome to my **Credit Card Fraud Detection** project where I use 4 different **machine learning models** in an attempt to predict fraudulent transactions.

Below is the list of the models that are used for this project:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **XGBoost** (spoiler: this one performed the best!)

The problem of fraud detection is tricky, as fraudulent transactions make up only **0.17%** of all transactions. This meant I needed to address class imbalance by using oversampling and undersampling techniques. My goal was to build a model that maximizes recall since missing fraud cases is worse than false alarms.

---

## Why I Chose This Project

I've always been interested in **data science and machine learning**, and fraud detection is such a cool real-world application. Every day, millions of people use credit cards, and fraudsters are always finding new ways to trick the system. I wanted to see whether I could build a model that would help **catch these fraudsters** and protect people's money.

This project was a huge learning experience for me since I had to deal with **imbalanced data**, apply **feature scaling**, try different **machine learning models**, and tune my approach to maximize **recall** since catching fraud is the #1 priority.

---

## Dataset

I used the Credit Card Fraud Detection Dataset from Kaggle. It contains **284,807 transactions** but only **492 (0.17%)** are frauds.

### Key Features:

- **Time** – Seconds elapsed between the transaction and the first transaction in the dataset.
- **Amount** – Transaction value.
- **V1 to V28** – Anonymized PCA-transformed features.
- **Class** – Target variable (0 = Legit, 1 = Fraud).

Fraudulent transactions are super rare; hence class imbalance was one big challenge here in this problem.

---

## Technologies Used

I used **Python** and some libraries such as:

- **Pandas, NumPy** – For data wrangling
- **Matplotlib, Seaborn** – For plotting and visualizing trends
- **Scikit-learn** – For machine learning models
- **Imbalanced-learn** – To balance the dataset
- **XGBoost** – Advanced Boosting Algorithm!

Now you’re ready to go!

---

## Exploratory Data Analysis

1. **Checking Data Information**

When I first loaded the dataset, I checked for **missing values, class imbalance, and feature distributions**.

```python
credit_card.info()
credit_card.describe()
```

→ No missing values! But the fraud cases were **way too few** so I need to tackle later.

2. **Visualizing Class Imbalance**

```python
sns.countplot(x='Class', data=credit_card)
```

Fraud cases were **drowned** by the majority class (legit transactions). This meant I needed to **resample the data** to prevent my model from being biased toward non-fraud transactions.

### Analyzing Transaction Amounts

```python
sns.histplot(credit_card['Amount'], bins=50)
```

In this part, I noticed that fraudulent transactions tend to be **small amounts.**

---

## Data Preprocessing

**1. Feature Scaling**

Since transaction **amounts** varied wildly, I applied **standardization**:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
credit_card['Amount'] = scaler.fit_transform(credit_card[['Amount']])
```

**2. Handling Class Imbalance**

To make sure my model **didn’t ignore fraud cases**, I used **oversampling** (duplicate fraud cases to balance the dataset) and **undersampling** (reduce the number of non-fraud cases).

```python
from imblearn.over_sampling import RandomOverSampler
sampler = RandomOverSampler()
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

**3. Splitting Data**

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

I used **80% training, 20% testing** to evaluate my model properly.

---

## Building the Models

I trained **four models** to compare their performance:

### 1. Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
```

### 2. K-Nearest Neighbors (KNN)

```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```

### 3. Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```

### 4. XGBoost

```python
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
```

---

## Evaluating Performance

I use:

- **Precision** – How many detected frauds are actually frauds?
- **Recall** – How many frauds were correctly detected?
- **F1-Score** – Balance between Precision & Recall
- **ROC-AUC Score** – Model’s ability to distinguish classes

However, I focused on **recall** because **missing fraud is worse than flagging legit transactions**.

```python
from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(y_test, y_pred))
print(f'ROC-AUC Score: {roc_auc_score(y_test, y_pred)}')
```

---

## Results

| Model | Sampling Type | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | Undersampling | 5.02% | 90.41% | 9.51% | 0.9374 |
| Logistic Regression | Oversampling | 6.53% | 89.73% | 12.17% | 0.9376 |
| Decision Tree | Undersampling | 4.21% | 87.67% | 8.04% | 0.9213 |
| Decision Tree | Oversampling | 8.41% | 86.30% | 15.32% | 0.9234 |
| KNN | Undersampling | 12.26% | 87.67% | 21.51% | 0.9329 |
| KNN | Oversampling | 57.14% | 84.93% | 68.32% | 0.9241 |
| XGBoost | Default | 95.12% | 80.14% | 86.99% | 0.9006 |
| XGBoost | Scaled Features | 90.41% | 84.20% | 87.21% | 0.9265 |
- **We can see that XGBoost with scaled features** achieved the best balance between precision, recall, and F1-Score and while other models performed well, their precision was significantly lower, making them less reliable for fraud detection.

---

## Final Thoughts

This project taught me so much about the following:

- Fraud detection is a highly imbalanced problem, resampling will be required
- Recall is more important than Accuracy to minimize undetected fraud cases.
- XGBoost gives the best performance, yielding high precision, recall, and AUC score.
