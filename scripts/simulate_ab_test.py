import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.stats.proportion import proportions_ztest

def main():
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "outputs" / "logistic_churn_predictions.csv"

    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_output_path = output_dir / "ab_test_summary.csv"
    customer_output_path = output_dir / "ab_test_customers.csv"
    plot_output_path = output_dir / "ab_test_result.png"

    # 캠페인(쿠폰 등) 1인당 비용 가정 - 임의 가정값이며 실제 캠페인 비용으로 교체 필요
    COST_PER_TREATMENT = 5000  # 단위: 원

    df = pd.read_csv(input_path)

    # 1단계: 실제 이탈 위험이 있는 고객만 후보로 (churn_probability >= 0.5)
    risk_threshold = 0.5
    candidates = df[df["churn_probability"] >= risk_threshold].copy()

    # 2단계: 위험군 안에서 expected_loss(가치) 상위 30%만 캠페인 타겟으로 선정
    threshold = candidates["expected_loss"].quantile(0.7)
    ab_df = candidates[candidates["expected_loss"] >= threshold].copy()

    print("위험군 후보 수:", len(candidates))
    print("고위험 고객 수:", len(ab_df))
    print("expected_loss threshold:", round(threshold, 2))
    print("타겟 고객의 총 expected_loss:", round(ab_df["expected_loss"].sum(), 2))

    # A/B 그룹 랜덤 배정
    np.random.seed(42)
    ab_df["group"] = np.random.choice(
        ["control", "treatment"],
        size=len(ab_df)
    )

    # treatment 그룹은 churn probability 8%p 감소 가정
    def simulate_outcome(row):
        p = row["churn_probability"]

        if row["group"] == "treatment":
            p = max(0, p - 0.08)

        return np.random.binomial(1, p)

    ab_df["simulated_churn"] = ab_df.apply(simulate_outcome, axis=1)

    # 그룹별 결과 요약
    summary = ab_df.groupby("group")["simulated_churn"].agg(["count", "mean", "sum"])
    summary.columns = ["customers", "churn_rate", "churn_count"]

    print("\n=== A/B Test Summary ===")
    print(summary)

    # Lift 계산
    control_rate = summary.loc["control", "churn_rate"]
    treatment_rate = summary.loc["treatment", "churn_rate"]

    absolute_lift = control_rate - treatment_rate
    relative_lift = absolute_lift / control_rate if control_rate != 0 else 0

    print("\n=== Lift ===")
    print(f"Control Churn Rate: {control_rate:.4f}")
    print(f"Treatment Churn Rate: {treatment_rate:.4f}")
    print(f"Absolute Lift: {absolute_lift:.4f}")
    print(f"Relative Lift: {relative_lift:.4f}")

    # 통계 검정
    success = [
        int(summary.loc["control", "churn_count"]),
        int(summary.loc["treatment", "churn_count"])
    ]

    nobs = [
        int(summary.loc["control", "customers"]),
        int(summary.loc["treatment", "customers"])
    ]

    stat, p_value = proportions_ztest(success, nobs)

    print("\n=== Statistical Test ===")
    print(f"Z-stat: {stat:.4f}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.05:
        print("통계적으로 유의미한 차이 있음")
    else:
        print("통계적으로 유의미하지 않음")

    # ROI 계산
    treatment_customers = int(summary.loc["treatment", "customers"])
    avg_clv_treatment = ab_df.loc[ab_df["group"] == "treatment", "clv"].mean()

    prevented_customers = absolute_lift * treatment_customers
    prevented_loss_value = prevented_customers * avg_clv_treatment
    total_cost = COST_PER_TREATMENT * treatment_customers
    net_benefit = prevented_loss_value - total_cost
    roi = net_benefit / total_cost if total_cost != 0 else 0

    print("\n=== ROI 분석 ===")
    print(f"예상 방어 고객 수: {prevented_customers:.1f}명")
    print(f"방어된 예상 손실액: {prevented_loss_value:,.0f}원")
    print(f"캠페인 총 비용: {total_cost:,.0f}원")
    print(f"순이익: {net_benefit:,.0f}원")
    print(f"ROI: {roi:.2%}")

    # 요약 저장
    summary_df = pd.DataFrame({
        "metric": [
            "high_risk_threshold",
            "control_churn_rate",
            "treatment_churn_rate",
            "absolute_lift",
            "relative_lift",
            "z_stat",
            "p_value",
            "cost_per_treatment",      
            "prevented_loss_value",   
            "total_cost",             
            "net_benefit",            
            "roi"                     
        ],
        "value": [
            threshold,
            control_rate,
            treatment_rate,
            absolute_lift,
            relative_lift,
            stat,
            p_value,
            COST_PER_TREATMENT,        
            prevented_loss_value,      
            total_cost,                
            net_benefit,               
            roi                        
        ]
    })

    summary_df.to_csv(summary_output_path, index=False, encoding="utf-8-sig")
    ab_df.to_csv(customer_output_path, index=False, encoding="utf-8-sig")

    print(f"\nA/B 테스트 요약 저장 완료: {summary_output_path}")
    print(f"A/B 테스트 고객 데이터 저장 완료: {customer_output_path}")

    # 시각화
    plot_df = summary.reset_index()

    plt.figure(figsize=(6, 4))
    plt.bar(plot_df["group"], plot_df["churn_rate"])
    plt.ylabel("Churn Rate")
    plt.title("A/B Test Result: Control vs Treatment")
    plt.ylim(0, max(plot_df["churn_rate"]) + 0.1)
    plt.savefig(plot_output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"그래프 저장 완료: {plot_output_path}")


if __name__ == "__main__":
    main()