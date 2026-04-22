import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_model_performance(output_dir: Path) -> None:
    models = ["V1 (Leakage)", "V2 (No Recency)", "V3 (Time-based)"]
    accuracy = [0.99, 0.74, 0.66]
    recall = [1.00, 0.67, 0.67]

    x = range(len(models))

    plt.figure(figsize=(8, 5))
    plt.plot(x, accuracy, marker="o", label="Accuracy")
    plt.plot(x, recall, marker="o", label="Recall")

    plt.xticks(list(x), models)
    plt.xlabel("Model Version")
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
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