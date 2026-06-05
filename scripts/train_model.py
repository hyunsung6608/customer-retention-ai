import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def main():
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / "processed" / "time_based_labeled_dataset.csv"

    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    logistic_output_path = output_dir / "logistic_churn_predictions.csv"
    rf_output_path = output_dir / "rf_churn_predictions.csv"

    df = pd.read_csv(input_path)

    # 사용할 feature 선택
    feature_cols = [
        "Recency",
        "Frequency",
        "Monetary",
        "customer_tenure",
        "avg_purchase_interval"
    ]

    X = df[feature_cols]
    customer_ids = df["CustomerID"]
    y = df["churn"]

    # 학습/테스트 데이터 분리
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, customer_ids, test_size=0.2, random_state=42, stratify=y
    )

    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # LogisticRegression 모델 학습
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    # 예측
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # 평가
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    # 예측 확률 확인
    result_df = X_test.copy()
    result_df["actual_churn"] = y_test.values
    result_df["predicted_churn"] = y_pred
    result_df["churn_probability"] = y_prob
    result_df["CustomerID"] = id_test.values

    result_df = result_df[
        [
            "CustomerID",
            "Recency",
            "Frequency",
            "Monetary",
            "customer_tenure",
            "avg_purchase_interval",
            "actual_churn",
            "predicted_churn",
            "churn_probability"
        ]
    ]

    result_df = result_df.sort_values(by="churn_probability", ascending=False)

    print("\nTop 10 High-Risk Customers:")
    print(result_df.head(10))

    # risk segment 분류
    result_df["risk_segment"] = pd.cut(
        result_df["churn_probability"],
        bins=[0, 0.5, 0.6, 0.7, 1.0],
        labels=["low", "medium", "high", "very_high"],
        include_lowest=True
    )

    result_df = result_df.sort_values(by="churn_probability", ascending=False)

    print("\nRisk Segment Distribution:")
    print(result_df["risk_segment"].value_counts())

    # 계수 확인
    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

    print("\nFeature Importance (Logistic Coefficients):")
    print(coef_df.sort_values(by="Coefficient", ascending=False))

    coef_plot_df = coef_df.reindex(
        coef_df["Coefficient"].abs().sort_values(ascending=True).index
    )

    plt.figure(figsize=(8, 5))
    plt.barh(coef_plot_df["Feature"], coef_plot_df["Coefficient"])
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.title("Feature Importance (Logistic Regression Coefficients)")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_logistic.png", dpi=300)
    plt.close()

    result_df.to_csv(logistic_output_path, index=False, encoding="utf-8-sig")
    print(f"\nLogistic 예측 결과 저장 완료: {logistic_output_path}")

    # Random Forest 모델 학습
    # 트리 기반 모델로 피처 스케일에 영향을 받지 않아 스케일링 없이 원본 데이터로 학습
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    rf_model.fit(X_train, y_train)

    # 예측
    rf_y_pred = rf_model.predict(X_test)
    rf_y_prob = rf_model.predict_proba(X_test)[:, 1]

    # 평가
    print("\n===== Random Forest =====")
    print("Accuracy:", accuracy_score(y_test, rf_y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, rf_y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, rf_y_pred))
    print("ROC-AUC: ", roc_auc_score(y_test, rf_y_prob))

    # Feature Importance
    rf_importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance (Random Forest):")
    print(rf_importance_df)

    # 확률 기준 상위 위험 고객
    rf_result_df = X_test.copy()
    rf_result_df["actual_churn"] = y_test.values
    rf_result_df["predicted_churn"] = rf_y_pred
    rf_result_df["churn_probability"] = rf_y_prob
    rf_result_df["CustomerID"] = id_test.values

    rf_result_df = rf_result_df[
        [
            "CustomerID",
            "Recency",
            "Frequency",
            "Monetary",
            "customer_tenure",
            "avg_purchase_interval",
            "actual_churn",
            "predicted_churn",
            "churn_probability"
        ]
    ]

    rf_result_df = rf_result_df.sort_values(by="churn_probability", ascending=False)

    print("\nTop 10 High-Risk Customers (Random Forest):")
    print(rf_result_df.head(10))

    rf_result_df["risk_segment"] = pd.cut(
        rf_result_df["churn_probability"],
        bins=[0, 0.5, 0.6, 0.7, 1.0],
        labels=["low", "medium", "high", "very_high"],
        include_lowest=True
    )

    print("\nRisk Segment Distribution (Random Forest):")
    print(rf_result_df["risk_segment"].value_counts())

    rf_result_df.to_csv(rf_output_path, index=False, encoding="utf-8-sig")
    print(f"\nRandom Forest 예측 결과 저장 완료: {rf_output_path}")


if __name__ == "__main__":
    main()