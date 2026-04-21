import pandas as pd
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent
    feature_path = project_root / "data" / "processed" / "past_customer_features.csv"
    future_path = project_root / "data" / "processed" / "future_data.csv"
    output_path = project_root / "data" / "processed" / "time_based_labeled_dataset.csv"

    snapshot_date = pd.Timestamp("2011-09-01")
    churn_window_end = snapshot_date + pd.Timedelta(days=90)

    features_df = pd.read_csv(feature_path)
    future_df = pd.read_csv(future_path)

    future_df["InvoiceDate"] = pd.to_datetime(future_df["InvoiceDate"])

    # snapshot 이후 90일 이내 구매한 고객 찾기
    active_future_customers = future_df[
        (future_df["InvoiceDate"] >= snapshot_date) &
        (future_df["InvoiceDate"] < churn_window_end)
    ]["CustomerID"].dropna().astype(int).unique()

    active_future_customers = set(active_future_customers)

    # churn label 생성
    features_df["churn"] = features_df["CustomerID"].apply(
        lambda customer_id: 0 if customer_id in active_future_customers else 1
    )

    features_df.to_csv(output_path, index=False)

    print("미래 기준 churn label 생성 완료")
    print(f"snapshot_date: {snapshot_date}")
    print(f"churn window end: {churn_window_end}")
    print(features_df[["CustomerID", "Recency", "Frequency", "Monetary", "churn"]].head())
    print("\nChurn distribution:")
    print(features_df["churn"].value_counts())
    print(f"\n저장 경로: {output_path}")


if __name__ == "__main__":
    main()