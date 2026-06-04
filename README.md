# 📊 AI-Powered Customer Retention Analytics

## 📌 Overview

Customer churn is one of the most costly problems in retail.
Acquiring a new customer costs 5–7x more than retaining 
an existing one, making early churn detection critical 
for business sustainability.

This project builds an end-to-end churn prediction pipeline 
— from raw transaction data to model-driven retention strategy 
— and validates its effectiveness through A/B test simulation.

Key highlight: During development, a data leakage issue 
(Accuracy ~0.99) was identified and resolved by redesigning 
the dataset using a time-based approach, resulting in a 
more realistic and reliable model.

---

## ⚡ Key Highlights

- Identified and resolved **data leakage** 
  (Accuracy 0.99 → 0.68) through time-based dataset redesign
- Selected **Logistic Regression** over Random Forest 
  for higher Recall on churn class (0.72 vs 0.54), 
  prioritizing detection over precision
- A/B simulation showed **22.2% churn reduction** 
  in treatment group (p=0.013), validating model-driven targeting
- Demonstrated full pipeline from raw data 
  to business decision validation

---

## 🎯 Objective

* Detect high-risk customers before they churn, 
  enabling proactive intervention
* Identify which behavioral signals best predict churn 
  (recency, frequency, purchase interval)
* Prioritize retention budget by risk tier 
  (Very High / High / Medium / Low)
* Validate that model-driven targeting reduces churn 
  vs. no-action baseline (A/B simulation)

---

## 📂 Dataset

* **Online Retail Dataset**
* Source: UCI Machine Learning Repository - Online Retail Dataset
* Transaction-level customer purchase data

The raw dataset is not included in this repository.

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
→ A/B Test Simulation
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

This confirmed that the model was relying on leakage rather than learning real patterns.

---

## 🔁 Solution: Time-Based Data Design

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

* Past Data → used for feature generation
* Future Data → used for churn labeling

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

Customers are labeled as churn if they do not purchase within 90 days after the snapshot date.

An earlier version of churn labeling (`create_churn_label.py`) was used in a baseline experiment that included data leakage.  
The final pipeline uses `create_future_churn_label.py` to generate time-based labels without leakage.

```
data/processed/time_based_labeled_dataset.csv
```

---

## 🤖 Model Training

### Models Used

* Logistic Regression
* Random Forest

---

## 📊 Model Comparison

| Version | Setup                                 | Accuracy | Recall (Churn) | Insight         |
| ------- | ------------------------------------- | -------- | -------------- | --------------- |
| V1      | Recency-based label + Recency feature | ~0.99    | ~1.00          | Data leakage    |
| V2      | Recency-based label without Recency   | ~0.74    | ~0.67          | Validation step |
| V3      | Time-based dataset                    | **0.68** | **0.72**       | Realistic model |

![Model Comparison](outputs/model_performance_comparison.png)

---

## 📈 Final Results

### Logistic Regression

* Accuracy: **0.68**
* Recall (churn): **0.72**
* ROC-AUC: **0.74**

### Random Forest

* Accuracy: **0.63**
* Recall (churn): **0.54**
* ROC-AUC: **0.69**

Logistic Regression was selected as the final model due to higher Recall on the churn class (0.72 vs 0.54), 
as missing a churning customer is more costly than 
a false alarm.

---

## 📈 Feature Insights

![Feature Importance](outputs/feature_importance_logistic.png)

* High Recency → higher churn risk
* Low Frequency → higher churn risk
* Short tenure → higher churn risk
* Long purchase interval → higher churn risk

Low engagement customers are more likely to churn.

---

## 🎯 Risk Segmentation

Customers are classified based on churn probability:

* 0.0–0.5 → Low Risk
* 0.5–0.6 → Medium Risk
* 0.6–0.7 → High Risk
* 0.7+ → Very High Risk

---

### Current Distribution (Test Set)

The following distribution is based on the logistic regression prediction results from the test set, not the full customer population.

* Low Risk: 321
* Medium Risk: 186
* High Risk: 122
* Very High Risk: 35

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

## 🧪 A/B Test Simulation

### Objective

To validate whether targeting high-risk customers with retention strategies can effectively reduce churn.

---

### Experiment Design

* Target: Top 30% customers by churn probability
* Control: no action
* Treatment: retention strategy applied (simulated)
* Assumption: churn probability reduced by **8%p**

---

### Result

| Group     | Customers | Churn Rate |
| --------- | --------: | ---------: |
| Control   |        99 |      70.7% |
| Treatment |       100 |      55.0% |

* Absolute Lift: **15.7%p decrease**
* Relative Improvement: **22.2% improvement**

---

### Statistical Significance

* p-value: 0.0219

The result is statistically significant (p < 0.05).

---

### Visualization

![A/B Test Result](outputs/ab_test_result.png)

---

### Interpretation

Retention strategies targeting high-risk customers can significantly reduce churn.

This project connects prediction results to real business decision-making validation.

---

### Note

This experiment is based on simulation, not real campaign data.

---

## 📁 Project Structure

Some directories (e.g., `notebooks/`, `models/`, `config/`) are reserved for future extensions such as experimentation, model persistence, and configuration management.

```
customer-retention-ai/
│
├── data/
│   ├── raw/
│   │   └── online_retail.csv   # manually downloaded, not committed
│   │
│   └── processed/
│       ├── cleaned_retail.csv
│       ├── customer_features.csv
│       ├── customer_features_labeled.csv
│       ├── past_data.csv
│       ├── future_data.csv
│       ├── past_customer_features.csv
│       └── time_based_labeled_dataset.csv
│
├── scripts/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── create_churn_label.py
│   ├── create_time_based_dataset.py
│   ├── create_past_features.py
│   ├── create_future_churn_label.py
│   ├── train_model.py
│   ├── simulate_ab_test.py
│   └── visualize_results.py
│
├── outputs/
│   ├── model_performance_comparison.png
│   ├── feature_importance_logistic.png
│   ├── logistic_churn_predictions.csv
│   ├── rf_churn_predictions.csv
│   ├── ab_test_customers.csv
│   ├── ab_test_summary.csv
│   └── ab_test_result.png
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

## 📥 Dataset Preparation

This project uses the **Online Retail Dataset** from the UCI Machine Learning Repository.

To run this project locally:

1. Download the Online Retail dataset from the UCI Machine Learning Repository.
2. Convert or save the dataset as `online_retail.csv`.
3. Place the file in the following path:

```text
data/raw/online_retail.csv
```

The raw dataset is not included in this repository because it is excluded by `.gitignore`.

```gitignore
data/raw/*.csv
```

This keeps the repository lightweight and avoids committing raw external data.

---

## 🧠 Key Takeaways

1. Data leakage can significantly distort model performance
2. Time-based dataset design is essential
3. Customer behavior is a strong predictor of churn
4. Machine learning outputs should be connected to business actions
5. A/B testing enables validation of data-driven strategies

---

## 🚀 Tech Stack

* Python (pandas, numpy)
* scikit-learn
* statsmodels
* matplotlib
* Git & GitHub

---

## 📌 Conclusion

This project demonstrates the transition from basic data analysis to a production-style machine learning pipeline.

It not only predicts churn but also validates retention strategies through simulation, bridging the gap between machine learning and real-world decision-making.

## 🚀 How to Run

### 1. Clone repository

```bash
git clone https://github.com/hyunsung6608/customer-retention-ai.git
cd customer-retention-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare dataset

Download the Online Retail dataset, save it as `online_retail.csv`, and place it in:

```text
data/raw/online_retail.csv
```

### 4. Run data preprocessing

```bash
python scripts/preprocess.py
```

### 5. Feature engineering

```bash
python scripts/feature_engineering.py
```

### 6. Create time-based dataset

```bash
python scripts/create_time_based_dataset.py
```

### 7. Create past features for leakage prevention

```bash
python scripts/create_past_features.py
```

### 8. Create churn label using future window

```bash
python scripts/create_future_churn_label.py
```

### 9. Train model

```bash
python scripts/train_model.py
```

### 10. Run A/B test simulation

```bash
python scripts/simulate_ab_test.py
```

### 11. Generate model comparison visualization

```bash
python scripts/visualize_results.py
```