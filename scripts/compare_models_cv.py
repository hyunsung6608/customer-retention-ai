import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def main():
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / "processed" / "time_based_labeled_dataset.csv"

    df = pd.read_csv(input_path)

    feature_cols = [
        "Recency",
        "Frequency",
        "Monetary",
        "customer_tenure",
        "avg_purchase_interval"
    ]

    X = df[feature_cols]
    y = df["churn"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic: 스케일링을 파이프라인 안에 넣어야 fold마다 train에만 fit됨 (스케일링 누수 방지)
    logistic_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42))
    ])

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=20,
        random_state=42,
        class_weight="balanced"
    )

    logistic_balanced_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42, class_weight="balanced"))
    ])
    
    models = {
        "Logistic Regression": logistic_pipeline,
        "Logistic Regression (balanced)": logistic_balanced_pipeline,
        "Random Forest": rf_model
    }

    for name, estimator in models.items():
        acc = cross_val_score(estimator, X, y, cv=cv, scoring="accuracy")
        rec = cross_val_score(estimator, X, y, cv=cv, scoring="recall")
        auc = cross_val_score(estimator, X, y, cv=cv, scoring="roc_auc")

        print(f"\n=== {name} (5-Fold CV) ===")
        print(f"Accuracy: {acc.mean():.4f} (+/- {acc.std():.4f})")
        print(f"Recall:   {rec.mean():.4f} (+/- {rec.std():.4f})")
        print(f"ROC-AUC:  {auc.mean():.4f} (+/- {auc.std():.4f})")


if __name__ == "__main__":
    main()