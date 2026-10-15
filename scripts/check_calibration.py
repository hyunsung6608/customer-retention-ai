import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, precision_score, recall_score, roc_auc_score
)

# 최종 모델 선택 근거 검증: LR(가중치 없음) vs LR(balanced)
# train_model.py 와 동일한 학습/테스트 분할을 사용한다.


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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "LR (가중치 없음)": LogisticRegression(random_state=42),
        "LR (balanced)": LogisticRegression(random_state=42, class_weight="balanced"),
    }

    probs = {}
    rows = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        probs[name] = y_prob
        rows.append({
            "model": name,
            "mean_pred_prob": y_prob.mean(),
            "actual_churn_rate": y_test.mean(),
            "brier_score": brier_score_loss(y_test, y_prob),
            "recall@0.5": recall_score(y_test, y_pred),
            "precision@0.5": precision_score(y_test, y_pred),
            "accuracy@0.5": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
        })

    result_df = pd.DataFrame(rows)
    print("=== 확률 보정 / 홀드아웃 성능 (threshold 0.5) ===")
    print(result_df.to_string(index=False))

    # 가중치 없는 LR 에서 balanced 의 홀드아웃 Recall 이상이 되는 가장 높은 threshold
    target_recall = result_df.loc[result_df["model"] == "LR (balanced)", "recall@0.5"].iloc[0]
    y_prob = probs["LR (가중치 없음)"]
    candidates = np.unique(y_prob)[::-1]
    for t in candidates:
        y_pred = (y_prob >= t).astype(int)
        if recall_score(y_test, y_pred) >= target_recall:
            break

    print("\n=== LR (가중치 없음) threshold 조정 ===")
    print(f"목표 Recall (LR balanced, threshold 0.5): {target_recall:.4f}")
    print(f"threshold: {t:.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    main()
