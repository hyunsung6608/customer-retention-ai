import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / "processed" / "customer_features_labeled.csv"

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
    y = df["churn"]

    # 학습/테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 모델 학습
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    # 예측
    y_pred = model.predict(X_test_scaled)

    # 평가
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 계수 확인
    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

    print("\nFeature Importance (Logistic Coefficients):")
    print(coef_df)


if __name__ == "__main__":
    main()