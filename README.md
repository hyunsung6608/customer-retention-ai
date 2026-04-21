# 📊 AI-Powered Customer Retention Analytics

## 📌 Overview

This project builds an end-to-end customer retention analytics system that predicts customer churn using transactional data and connects predictions to actionable retention strategies.

Starting from basic RFM analysis, the project evolves into a machine learning pipeline and addresses a critical issue in modeling — **data leakage** — by redesigning the dataset using a time-based approach.

---

## 🎯 Objective

* Predict customers who are likely to churn
* Identify key behavioral drivers of churn
* Segment customers based on risk levels
* Design retention strategies based on model outputs

---

## 📂 Dataset

* **Online Retail Dataset**
* Transaction-level customer purchase data

---

## 🏗️ Project Pipeline

```text
Raw Data
→ Data Preprocessing
→ Feature Engineering
→ Churn Labeling
→ Model Training
→ Churn Prediction
→ Risk Segmentation
→ Retention Strategy
```

---

## ⚙️ Data Processing

### 1. Data Preprocessing

* Removed missing CustomerID
* Filtered invalid transactions (Quantity ≤ 0, UnitPrice ≤ 0)
* Converted InvoiceDate to datetime
* Created Sales feature

**Output:**

```
data/processed/cleaned_retail.csv
```

---

### 2. Feature Engineering

Customer-level features were created:

* Recency
* Frequency
* Monetary
* customer_tenure
* avg_purchase_interval

**Output:**

```
data/processed/customer_features.csv
```

---

## ⚠️ Initial Approach & Problem (Data Leakage)

Churn was initially defined as:

```
Recency > 90 → churn = 1
```

However, **Recency was also used as an input feature**, leading to data leakage.

### Result:

* Accuracy ≈ 0.99 ~ 1.00
* Unrealistically perfect predictions

To validate this, Recency was removed and the model was retrained:

* Accuracy dropped to ~0.74

👉 This confirmed that the model was relying on leakage rather than learning real patterns.

---

## 🔁 Solution: Time-Based Data Design

To resolve data leakage, the dataset was redesigned to reflect a real-world prediction scenario.

### Key Idea

```
Past data → Feature generation  
Future data → Churn labeling
```

---

### Step 1: Snapshot Definition

```
snapshot_date = 2011-09-01
```

---

### Step 2: Data Split

* **Past Data** → used for feature generation
* **Future Data** → used for churn labeling

```
data/processed/past_data.csv
data/processed/future_data.csv
```

---

### Step 3: Feature Generation (Past Only)

```
data/processed/past_customer_features.csv
```

---

### Step 4: Future-Based Churn Label

Customers are labeled as churn if they **do not purchase within 90 days after the snapshot date**.

```
data/processed/time_based_labeled_dataset.csv
```

---

## 🤖 Model Training

### Models Used

* Logistic Regression (baseline)
* Random Forest (comparison)

---

## 📊 Model Comparison

| Version | Setup                                 | Accuracy | Recall (Churn) | Insight         |
| ------- | ------------------------------------- | -------- | -------------- | --------------- |
| V1      | Recency-based label + Recency feature | ~0.99    | ~1.00          | Data leakage    |
| V2      | Recency-based label without Recency   | ~0.74    | ~0.67          | Validation step |
| V3      | Time-based dataset (final)            | **0.66** | **0.67**       | Realistic model |

The performance trend clearly shows the impact of data leakage and the effectiveness of the time-based redesign.

![Model Comparison](outputs/model_performance_comparison.png)

---

## 📈 Final Results (Time-Based Model)

### Logistic Regression

* Accuracy: **0.66**
* Recall (churn): **0.67**
* ROC-AUC: **0.72**

### Random Forest

* Accuracy: 0.62
* Recall (churn): 0.52

👉 Logistic Regression was selected as the final model.

👉 The ROC-AUC score of 0.72 indicates that the model has a reasonable ability to distinguish between churn and non-churn customers.

👉 Although the overall accuracy is lower than the initial model, this reflects a more realistic performance after removing data leakage and redesigning the dataset using a time-based approach.

---

## 📈 Feature Insights

![Feature Importance](outputs/feature_importance_logistic.png)

The model shows that customer behavior patterns are strong indicators of churn:

* Customers with higher **Recency** (long inactivity) are more likely to churn
* Customers with lower **Frequency** are more likely to churn
* Customers with shorter **customer tenure** tend to churn more
* Customers with longer **purchase intervals** have higher churn risk

👉 **Low engagement customers are more likely to churn**

---

## 🎯 Risk Segmentation

Customers are segmented based on churn probability:

* High Risk
* Medium Risk
* Low Risk

---

## 💡 Retention Strategy

### High-Risk Customers

* Discount coupons
* Personalized recommendations

### Medium-Risk Customers

* Reminder emails
* Product suggestions

### Low-Risk Customers

* General engagement strategies

---

## 📁 Project Structure

```
customer-retention-ai/
│
├── data/
│   ├── raw/
│   │   └── online_retail.csv
│   │
│   └── processed/
│       ├── cleaned_retail.csv
│       ├── customer_features.csv
│       ├── past_data.csv
│       ├── future_data.csv
│       ├── past_customer_features.csv
│       └── time_based_labeled_dataset.csv
│
├── scripts/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── create_time_based_dataset.py
│   ├── create_past_features.py
│   ├── create_future_churn_label.py
│   ├── train_model.py
│   └── visualize_results.py
│
├── outputs/
│   ├── model_performance_comparison.png
│   └── feature_importance_logistic.png
│
├── notebooks/
├── models/
├── config/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🧠 Key Takeaways

1. Data leakage can significantly distort model performance
2. Time-based dataset design is essential for realistic prediction
3. Customer engagement metrics are strong predictors of churn
4. Machine learning outputs should be connected to business actions

---

## 🚀 Tech Stack

* Python (pandas, numpy)
* scikit-learn
* Git & GitHub

---

## 📌 Conclusion

This project demonstrates the transition from basic data analysis to a production-style machine learning pipeline.

By identifying and resolving data leakage through time-based dataset design, the model was improved to reflect a real-world churn prediction scenario and successfully connected to actionable retention strategies.
