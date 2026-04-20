import pandas as pd
from pathlib import Path


def main():
    # 경로 설정
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / "raw" / "online_retail.csv"
    output_path = project_root / "data" / "processed" / "cleaned_retail.csv"

    # 데이터 불러오기
    df = pd.read_csv(input_path, encoding="ISO-8859-1")

    print(f"원본 데이터 크기: {df.shape}")

    # 날짜형 변환
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # 결측 제거
    df = df.dropna(subset=["CustomerID"])

    # 이상치 제거
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # 타입 변환
    df["CustomerID"] = df["CustomerID"].astype(int)

    # 매출 컬럼 생성
    df["Sales"] = df["Quantity"] * df["UnitPrice"]

    print(f"전처리 후 데이터 크기: {df.shape}")

    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"전처리 완료 파일 저장: {output_path}")


if __name__ == "__main__":
    main()