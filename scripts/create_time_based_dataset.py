import pandas as pd
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / "processed" / "cleaned_retail.csv"
    output_dir = project_root / "data" / "processed"

    df = pd.read_csv(input_path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    snapshot_date = pd.Timestamp("2011-09-01")

    past_df = df[df["InvoiceDate"] < snapshot_date].copy()
    future_df = df[df["InvoiceDate"] >= snapshot_date].copy()

    past_output = output_dir / "past_data.csv"
    future_output = output_dir / "future_data.csv"

    past_df.to_csv(past_output, index=False)
    future_df.to_csv(future_output, index=False)

    print("데이터 분리 완료")
    print(f"snapshot_date: {snapshot_date}")
    print(f"과거 데이터 크기: {past_df.shape}")
    print(f"미래 데이터 크기: {future_df.shape}")
    print(f"저장 경로: {past_output}")
    print(f"저장 경로: {future_output}")


if __name__ == "__main__":
    main()