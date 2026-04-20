import pandas as pd
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / "processed" / "cleaned_retail.csv"
    output_path = project_root / "data" / "processed" / "customer_features.csv"

    # 전처리된 데이터 불러오기
    df = pd.read_csv(input_path)

    # 날짜형 변환
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # snapshot date 정의
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # 고객별 feature 생성
    customer_features = df.groupby("CustomerID").agg(
        first_purchase_date=("InvoiceDate", "min"),
        last_purchase_date=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Sales", "sum")
    ).reset_index()

    # Recency
    customer_features["Recency"] = (
        snapshot_date - customer_features["last_purchase_date"]
    ).dt.days

    # customer_tenure
    customer_features["customer_tenure"] = (
        customer_features["last_purchase_date"] - customer_features["first_purchase_date"]
    ).dt.days

    # avg_purchase_interval
    customer_features["avg_purchase_interval"] = customer_features.apply(
        lambda row: row["customer_tenure"] / (row["Frequency"] - 1)
        if row["Frequency"] > 1 else 0,
        axis=1
    )

    # 컬럼 순서 정리
    customer_features = customer_features[
        [
            "CustomerID",
            "first_purchase_date",
            "last_purchase_date",
            "Recency",
            "Frequency",
            "Monetary",
            "customer_tenure",
            "avg_purchase_interval",
        ]
    ]

    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    customer_features.to_csv(output_path, index=False)

    print("Customer feature 생성 완료")
    print(customer_features.head())
    print(f"저장 경로: {output_path}")


if __name__ == "__main__":
    main()