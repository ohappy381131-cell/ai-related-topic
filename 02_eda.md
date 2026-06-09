# EDA (Exploratory Data Analysis)

## EDA란 무엇인가

EDA(Exploratory Data Analysis, 탐색적 데이터 분석)는 1977년 통계학자 존 터키(John Tukey)가 제창한 개념으로, 본격적인 모델링이나 가설 검증에 앞서 **데이터 자체를 열어보고 구조를 이해하는 과정**입니다.

한마디로 정의하면 이렇습니다:

> "데이터에게 '너 어떻게 생겼어?'라고 묻는 과정"

모델을 만들기 전에 데이터의 분포, 관계, 품질 문제를 먼저 직접 확인하지 않으면, 이후에 만드는 모든 모델은 모래 위의 성입니다.

---

## 5단계 상세 설명

### 1단계 — 데이터 파악 (Data Understanding)

데이터를 처음 받았을 때 가장 먼저 하는 일입니다.

```python
df.shape        # 행(샘플) × 열(피처) 크기
df.dtypes       # 각 컬럼의 자료형 (int, float, object 등)
df.head(5)      # 상위 5행 확인
df.info()       # 메모리, 널 여부, 타입 한눈에
df.describe()   # 수치형 컬럼의 기술통계 (평균, 표준편차, min/max 등)
```

이 단계에서 답해야 할 질문은 "샘플이 몇 개고, 피처(feature)가 몇 개인가? 수치형(numerical)인가 범주형(categorical)인가?"입니다.

---

### 2단계 — 결측치·이상치 탐지 (Missing Values & Outliers)

현실 데이터에는 반드시 문제가 있습니다. 이 단계에서 그 문제를 정량화합니다.

**결측치(Missing Values) 처리:**

```python
df.isnull().sum()         # 컬럼별 결측 개수
df.isnull().mean() * 100  # 결측 비율(%)
```

결측 비율에 따라 제거(drop), 대체(imputation — 평균, 중앙값, 최빈값), 또는 별도 플래그(indicator) 피처 생성 중 전략을 결정합니다.

**이상치(Outliers) 탐지:**

- IQR(사분위수 범위, Interquartile Range) 방법: $Q_1 - 1.5 \times IQR \sim Q_3 + 1.5 \times IQR$ 바깥 값
- z-score 방법: 평균에서 $3\sigma$ 이상 벗어난 값
- 시각적으로는 박스플롯(boxplot)이 가장 직관적

---

### 3단계 — 단변량 분석 (Univariate Analysis)

변수를 하나씩 독립적으로 분석합니다. 각 피처가 어떻게 분포(distribution)되어 있는지 파악하는 것이 목표입니다.

수치형 변수의 경우 히스토그램(histogram)과 커널 밀도 추정(KDE, Kernel Density Estimation) 플롯으로 분포 형태를 봅니다. 이때 왜도(skewness, 분포의 비대칭)와 첨도(kurtosis, 분포의 뾰족함)를 확인합니다. 오른쪽으로 긴 꼬리(right-skewed)인지, 정규분포에 가까운지가 나중에 어떤 모델을 쓸지에 영향을 미칩니다.

범주형 변수의 경우 `value_counts()`로 클래스 비율을 확인합니다. 클래스 불균형(class imbalance)이 있다면 모델 학습 전 대응 전략이 필요합니다.

---

### 4단계 — 이변량·다변량 분석 (Bivariate / Multivariate Analysis)

피처들 사이의 관계를 분석합니다. 핵심 질문은 "어떤 피처가 타깃(target) 변수와 관련이 있는가?"입니다.

수치형 vs 수치형: 산점도(scatter plot)와 피어슨 상관계수(Pearson correlation coefficient, $-1 \sim +1$)로 선형 관계를 봅니다. 이것을 행렬로 정리한 것이 상관행렬(correlation matrix)이고, 히트맵(heatmap)으로 시각화합니다.

다변량으로는 `pairplot`으로 모든 변수 쌍(pair)의 관계를 한 화면에서 파악합니다.

---

### 5단계 — 인사이트 도출 & 피처 후보 선정

앞 4단계의 관찰을 바탕으로 가설(hypothesis)을 수립하고, 어떤 피처를 모델에 넣을지 결정하고, 어떤 전처리(feature engineering, normalization, encoding)가 필요한지 방향을 잡습니다.

---

## EDA가 ML/DL에서 갖는 진짜 의미

EDA는 "데이터 정리"가 아닙니다. 정확히는 **도메인 지식(domain knowledge)과 데이터 신호(signal)가 처음으로 만나는 지점**입니다. 좋은 EDA는 이미 분석의 절반입니다.

| 단계 | 주요 도구 | 핵심 질문 |
|---|---|---|
| 데이터 파악 | `shape`, `info`, `describe` | 데이터가 어떻게 생겼나? |
| 결측·이상치 | `isnull`, IQR, z-score | 품질 문제가 얼마나 있나? |
| 단변량 분석 | histogram, KDE | 각 변수는 어떻게 분포하나? |
| 이변량 분석 | heatmap, scatter | 변수들 사이에 어떤 관계가 있나? |
| 인사이트 | pairplot, 가설 수립 | 무엇이 타깃과 관련이 있나? |
