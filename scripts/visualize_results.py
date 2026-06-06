import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_model_performance(output_dir: Path) -> None:
    # V1, V2 : 데이터 누수 실험 기록값 (별도 스크립트 없음, 재현 불가)
    # V3 : model_metrics.csv에서 읽어옴

    v1_accuracy, v1_recall = 0.99, 1.00
    v2_accuracy, v2_recall = 0.74, 0.67

    metrics_path = output_dir / "model_metrics.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"model_metrics.csv not found at {metrics_path}\n"
            "train_model.py를 먼저 실행해 주세요: python scripts/train_model.py"
        )

    metrics_df = pd.read_csv(metrics_path)
    lr_row = metrics_df[metrics_df["model"] == "logistic"].iloc[0]

    v3_accuracy = lr_row["accuracy"]
    v3_recall = lr_row["recall_churn"]


    models = ["V1 (Leakage)", "V2 (No Recency)", "V3 (Time-based)"]
    accuracy = [v1_accuracy, v2_accuracy, v3_accuracy]
    recall = [v1_recall, v2_recall, v3_recall]

    x = range(len(models))

    plt.figure(figsize=(8, 5))
    plt.plot(x, accuracy, marker="o", label="Accuracy")
    plt.plot(x, recall, marker="o", label="Recall")

    plt.xticks(list(x), models)
    plt.xlabel("Model Version")
    plt.ylabel("Score")
    plt.title("Model Version Comparison (Experiment Summary)")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()

    save_path = output_dir / "model_performance_comparison.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_model_performance(output_dir)

    print("All visualizations created successfully.")


if __name__ == "__main__":
    main()