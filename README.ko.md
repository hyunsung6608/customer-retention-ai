# 📊 AI 기반 고객 이탈 예측 분석

## 📌 프로젝트 개요

고객 이탈은 리테일 비즈니스에서 가장 큰 비용 문제 중 하나입니다.
신규 고객을 유치하는 비용은 기존 고객을 유지하는 것보다 5~7배 더 많이 듭니다.
따라서 이탈 징후를 사전에 감지하는 것이 비즈니스 지속성에 매우 중요합니다.

이 프로젝트는 원시 트랜잭션 데이터에서 시작해 모델 기반 리텐션 전략까지 이어지는
end-to-end 이탈 예측 파이프라인을 구축하고, A/B 테스트 시뮬레이션을 통해 효과를 검증합니다.

**핵심 포인트:** 개발 과정에서 데이터 누수(Data Leakage) 문제(정확도 ~0.99)를
직접 발견하고, 시계열 기반 데이터셋 재설계를 통해 해결했습니다.

---

## ⚡ 핵심 성과

- **데이터 누수 발견 및 해결** — 정확도 0.99 → 0.68로 재설계 (시계열 기반)
- **Logistic Regression 채택** — Random Forest 대비 이탈 클래스 Recall 우위 (0.72 vs 0.54), 미탐지 비용을 우선 고려
- **A/B 시뮬레이션 결과** — 처리 그룹에서 이탈률 22.2% 감소 (p=0.0219)
- 원시 데이터에서 비즈니스 의사결정 검증까지 전체 파이프라인 구현

---

## 🎯 분석 목적

* 이탈 가능성이 높은 고객을 사전에 탐지하여 선제적 개입 가능하도록 함
* 이탈 예측에 가장 유효한 행동 신호 파악 (최근성, 구매 빈도, 구매 간격 등)
* 위험 등급별 리텐션 예산 우선순위 설정 (매우 높음 / 높음 / 보통 / 낮음)
* 모델 기반 타겟팅이 미조치 대비 이탈률을 실제로 낮추는지 A/B 시뮬레이션으로 검증

---

## 📂 데이터셋

* **Online Retail Dataset**
* 출처: UCI Machine Learning Repository
* 트랜잭션 단위 고객 구매 데이터

원본 데이터셋은 이 저장소에 포함되어 있지 않습니다.

---

## 🏗️ 프로젝트 파이프라인

```text
원시 데이터
→ 데이터 전처리
→ 피처 엔지니어링
→ 이탈 레이블 생성
→ 모델 학습
→ 이탈 예측
→ 위험 세그멘테이션
→ 리텐션 전략 수립
→ A/B 테스트 시뮬레이션
```

---

## ⚙️ 데이터 처리

### 1. 데이터 전처리

* CustomerID 누락 행 제거
* 유효하지 않은 트랜잭션 필터링 (Quantity ≤ 0, UnitPrice ≤ 0)
* InvoiceDate를 datetime 형식으로 변환
* Sales 피처 생성

**출력 파일:**

```
data/processed/cleaned_retail.csv
```

---

### 2. 피처 엔지니어링

고객 단위 피처를 생성했습니다:

* Recency (최근성)
* Frequency (구매 빈도)
* Monetary (구매 금액)
* customer_tenure (고객 기간)
* avg_purchase_interval (평균 구매 간격)

**출력 파일:**

```
data/processed/customer_features.csv
```

---

## ⚠️ 초기 접근 방식과 문제점 (데이터 누수)

초기 이탈 정의:

```
Recency > 90 → churn = 1
```

그런데 **Recency가 입력 피처로도 사용**되어 데이터 누수가 발생했습니다.

### 결과:

* 정확도 ≈ 0.99 ~ 1.00
* 비현실적으로 완벽한 예측

검증을 위해 Recency를 피처에서 제거하고 재학습:

* 정확도가 ~0.74로 하락

모델이 실제 패턴을 학습한 것이 아니라 누수에 의존하고 있었음을 확인했습니다.

---

## 🔁 해결 방법: 시계열 기반 데이터 설계

### 핵심 아이디어

```
과거 데이터 → 피처 생성  
미래 데이터 → 이탈 레이블 생성
```

---

### Step 1: 스냅샷 날짜 정의

```
snapshot_date = 2011-09-01
```

---

### Step 2: 데이터 분리

* 과거 데이터 → 피처 생성에 사용
* 미래 데이터 → 이탈 레이블 생성에 사용

```
data/processed/past_data.csv
data/processed/future_data.csv
```

---

### Step 3: 과거 데이터 기반 피처 생성

```
data/processed/past_customer_features.csv
```

---

### Step 4: 미래 기반 이탈 레이블 생성

스냅샷 날짜 이후 90일 내에 구매가 없는 고객을 이탈로 정의합니다.

초기 버전(`create_churn_label.py`)은 데이터 누수가 포함된 베이스라인 실험에 사용되었습니다.
최종 파이프라인은 `create_future_churn_label.py`를 사용하여 누수 없는 시계열 레이블을 생성합니다.

```
data/processed/time_based_labeled_dataset.csv
```

---

## 🤖 모델 학습

### 사용 모델

* Logistic Regression : StandaradScaler 적용 (거리 기반 모델)
* Random Forest : 스케일링 없음 (트리기반 모델, 스케일 불변)

---

## 📊 모델 비교

| 버전 | 설정                                  | 정확도   | Recall (이탈) | 비고            |
| ---- | ------------------------------------- | -------- | ------------- | --------------- |
| V1   | Recency 기반 레이블 + Recency 피처    | ~0.99    | ~1.00         | 데이터 누수     |
| V2   | Recency 기반 레이블, Recency 피처 제거 | ~0.74    | ~0.67         | 검증 단계       |
| V3   | 시계열 기반 데이터셋                  | **0.68** | **0.72**      | 현실적인 모델   |

![모델 비교](outputs/model_performance_comparison.png)

---

## 📈 최종 결과

### Logistic Regression

* 정확도: **0.68**
* Recall (이탈): **0.72**
* ROC-AUC: **0.74**

### Random Forest

* 정확도: **0.63**
* Recall (이탈): **0.54**
* ROC-AUC: **0.69**

이탈 클래스 Recall이 더 높은 Logistic Regression을 최종 모델로 선택했습니다 (0.72 vs 0.54).
이탈 고객을 놓치는 비용이 정상 고객을 잘못 분류하는 비용보다 크기 때문입니다.

---

## 📈 피처 중요도 인사이트

![피처 중요도](outputs/feature_importance_logistic.png)

* Recency 높음 → 이탈 위험 높음
* Frequency 낮음 → 이탈 위험 높음
* 고객 기간 짧음 → 이탈 위험 높음
* 평균 구매 간격 김 → 이탈 위험 높음

참여도가 낮은 고객일수록 이탈 가능성이 높습니다.

---

## 🎯 위험 세그멘테이션

이탈 확률에 따라 고객을 분류합니다:

* 0.0–0.5 → 낮은 위험
* 0.5–0.6 → 보통 위험
* 0.6–0.7 → 높은 위험
* 0.7+ → 매우 높은 위험

---

### 현재 분포 (테스트 세트 기준)

전체 고객 모집단이 아닌 테스트 세트의 Logistic Regression 예측 결과 기준입니다.

* 낮은 위험: 321명
* 보통 위험: 186명
* 높은 위험: 122명
* 매우 높은 위험: 35명

---

## 💡 리텐션 전략

### 고위험 고객

* 할인 쿠폰 제공
* 개인화 추천

### 보통 위험 고객

* 리마인더 이메일
* 상품 제안

### 저위험 고객

* 일반 참여 유도 전략

---

## 🧪 A/B 테스트 시뮬레이션

### 목적

고위험 고객을 대상으로 한 리텐션 전략이 이탈률을 효과적으로 낮출 수 있는지 검증합니다.

---

### 실험 설계

* 대상: 이탈 확률 상위 30% 고객
* 대조군: 미조치
* 처리군: 리텐션 전략 적용 (시뮬레이션)
* 가정: 이탈 확률 **8%p** 감소

---

### 결과

| 그룹   | 고객 수 | 이탈률 |
| ------ | ------: | -----: |
| 대조군 |      99 |  70.7% |
| 처리군 |     100 |  55.0% |

* 절대적 개선: **15.7%p 감소**
* 상대적 개선: **22.2% 향상**

---

### 통계적 유의성

* p-value: 0.0219

결과는 통계적으로 유의합니다 (p < 0.05).

---

### 시각화

![A/B 테스트 결과](outputs/ab_test_result.png)

---

### 해석

고위험 고객을 대상으로 한 리텐션 전략은 이탈률을 유의미하게 낮출 수 있습니다.

이 프로젝트는 예측 결과를 실제 비즈니스 의사결정 검증으로 연결합니다.

---

### 주의사항

이 실험은 실제 캠페인 데이터가 아닌 시뮬레이션 기반입니다.

---

## 📁 프로젝트 구조

`notebooks/`, `models/`, `config/` 디렉토리는 향후 실험, 모델 저장, 설정 관리 등을 위해 예약되어 있습니다.

```
customer-retention-ai/
│
├── data/
│   ├── raw/
│   │   └── online_retail.csv   # 직접 다운로드 필요, 커밋되지 않음
│   │
│   └── processed/
│       ├── cleaned_retail.csv
│       ├── customer_features.csv
│       ├── customer_features_labeled.csv
│       ├── past_data.csv
│       ├── future_data.csv
│       ├── past_customer_features.csv
│       └── time_based_labeled_dataset.csv
│
├── scripts/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── create_churn_label.py
│   ├── create_time_based_dataset.py
│   ├── create_past_features.py
│   ├── create_future_churn_label.py
│   ├── train_model.py
│   ├── simulate_ab_test.py
│   └── visualize_results.py
│
├── outputs/
│   ├── model_performance_comparison.png
│   ├── feature_importance_logistic.png
│   ├── logistic_churn_predictions.csv
│   ├── rf_churn_predictions.csv
│   ├── ab_test_customers.csv
│   ├── ab_test_summary.csv
│   └── ab_test_result.png
│
├── notebooks/
├── models/
├── config/
│
├── README.md
├── README.ko.md
├── requirements.txt
└── .gitignore
```

---

## 📥 데이터셋 준비

이 프로젝트는 UCI Machine Learning Repository의 **Online Retail Dataset**을 사용합니다.

로컬에서 실행하려면:

1. UCI Machine Learning Repository에서 Online Retail 데이터셋을 다운로드합니다.
2. 파일명을 `online_retail.csv`로 저장합니다.
3. 아래 경로에 파일을 위치시킵니다:

```text
data/raw/online_retail.csv
```

원본 데이터셋은 `.gitignore`에 의해 저장소에 포함되지 않습니다.

---

## 🧠 핵심 배운 점

1. 데이터 누수는 모델 성능을 심각하게 왜곡할 수 있다 — 정확도 0.99가 나왔을 때 항상 의심하라
2. 피처와 레이블은 같은 시간 범위를 공유하면 안 된다 — 스냅샷 기반 설계로 미래 데이터 누수를 방지
3. Recency와 avg_purchase_interval이 가장 강한 신호 — 이탈 전 고객은 명확한 행동 패턴을 보인다
4. 머신러닝 결과는 비즈니스 액션으로 연결되어야 한다
5. A/B 테스트는 데이터 기반 전략의 효과를 검증하는 핵심 도구다

---

## 🚀 기술 스택

* Python (pandas, numpy)
* scikit-learn
* statsmodels
* matplotlib
* Git & GitHub

---

## 📌 결론

이 프로젝트는 기초 데이터 분석에서 프로덕션 수준의 머신러닝 파이프라인으로의 전환을 보여줍니다.

단순한 이탈 예측을 넘어 시뮬레이션을 통한 리텐션 전략 검증까지 수행하며,
머신러닝과 실제 비즈니스 의사결정 사이의 간극을 연결합니다.

## 🚀 실행 방법

### 1. 저장소 클론

```bash
git clone https://github.com/hyunsung6608/customer-retention-ai.git
cd customer-retention-ai
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 데이터셋 준비

Online Retail 데이터셋을 다운로드하여 `online_retail.csv`로 저장 후 아래 경로에 위치:

```text
data/raw/online_retail.csv
```

### 4. 데이터 전처리 실행

```bash
python scripts/preprocess.py
```

### 5. 시계열 기반 데이터셋 생성

```bash
python scripts/create_time_based_dataset.py
```

### 6. 누수 방지를 위한 과거 피처 생성

```bash
python scripts/create_past_features.py
```

### 7. 미래 윈도우 기반 이탈 레이블 생성

```bash
python scripts/create_future_churn_label.py
```

### 8. 모델 학습

```bash
python scripts/train_model.py
```

### 9. A/B 테스트 시뮬레이션 실행

```bash
python scripts/simulate_ab_test.py
```

### 10. 모델 비교 시각화 생성

```bash
python scripts/visualize_results.py
```

## 🔬 데이터 누수 실험 재현 (선택사항)
아래 스크립트는 최종 파이프라인과 무관합니다.
데이터 누수 섹션에서 설명한 초기 실험을 직접 재현해보고 싶을 때 사용합니다.

```bash
# Step 1: 전체 데이터셋 기반 고객 피처 생성
python scripts/feature_engineering.py

# Step 2: Recency 임계값 기반 이탈 레이블 생성 (누수 포함)
python scripts/create_churn_label.py
```

⚠️ 이 스크립트들은 Recency를 피처와 레이블 생성 기준으로 동시에 사용하여 데이터 누수가 발생하고 정확도가 비현실적으로 높게 (~0.99) 나옵니다.
이 결과물을 모델 학습에 사용하지 마세요.