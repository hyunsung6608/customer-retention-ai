import pandas as pd
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / "processed" / "customer_features.csv"
    output_path = project_root / "data" / "processed" / "customer_features_labeled.csv"

    df = pd.read_csv(input_path)

    # churn label 생성
    df["churn"] = (df["Recency"] > 90).astype(int)

    print("Churn label 생성 완료")
    print(df[["CustomerID", "Recency", "churn"]].head())
    print(df["churn"].value_counts())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"저장 경로: {output_path}")


if __name__ == "__main__":
    main()