# 데이터 전처리: 결측값 · 중복값 · 이상치

---

## 1. 결측값(Missing Values) 처리

### 왜 문제인가

머신러닝 알고리즘 대부분은 NaN(Not a Number)을 처리하지 못합니다. scikit-learn의 모든 모델은 결측값이 있으면 에러를 냅니다. 그러므로 결측값은 반드시 처리해야 합니다.

### 처리 전략 결정 기준: 결측 비율

결측 비율(missing rate)에 따라 전략이 달라집니다.

| 결측 비율 | 권장 전략 |
|---|---|
| < 5% | `df.dropna()` — 행 제거, 정보 손실 미미 |
| 5 ~ 50% | 대체(imputation) |
| > 50% | 컬럼 자체 제거 (`df.drop(columns=...)`) |

### 대체(Imputation) 전략 상세

**수치형(numerical) 변수:**

- 이상치 없고 분포가 대칭(정규분포에 가까울 때) → **평균(mean)**
- 이상치 있거나 분포가 비대칭(skewed) → **중앙값(median)** — 이상치에 로버스트(robust)

```python
df['age'].fillna(df['age'].median(), inplace=True)
```

**범주형(categorical) 변수:**

- 최빈값(mode)으로 대체
- 결측 자체가 의미 있다면 `'Unknown'`이라는 새 카테고리 생성

**고급 기법:**

- **KNN Imputer**: $k$-최근접 이웃 기반 대체, 변수 간 관계를 반영
- **MICE** (Multiple Imputation by Chained Equations): 연쇄 방정식 다중 대체

### 결측 지시자(Missing Indicator) 피처

결측 패턴 자체가 타깃과 상관이 있을 때 유용한 기법입니다.

```python
df['income_missing'] = df['income'].isnull().astype(int)
```

예: 소득 항목을 일부러 안 채우는 패턴이 신용 위험과 연관될 수 있습니다.

---

## 2. 중복값(Duplicates) 처리

### 왜 문제인가

중복 데이터는 모델이 특정 패턴을 과도하게 학습하게 만들고(과적합, overfitting), 평가 지표를 왜곡합니다. 특히 같은 행이 train/test 양쪽에 들어가면 데이터 누수(data leakage)가 발생합니다.

### 완전 중복(Exact Duplicates) vs 부분 중복(Partial Duplicates)

**완전 중복** — 모든 컬럼 값이 동일한 행. 대부분 데이터 수집 오류이므로 제거합니다.

```python
df.duplicated().sum()            # 중복 행 개수 확인
df.drop_duplicates(keep='first') # 첫 번째만 남기고 제거
```

**부분 중복** — 특정 컬럼 기준으로만 중복인 경우. 도메인 지식(domain knowledge)으로 판단해야 합니다.

```python
df.duplicated(subset=['user_id', 'timestamp']).sum()
```

집계(aggregation)가 필요한 경우:

```python
df.groupby('user_id').agg({'amount': 'sum', 'visit': 'count'}).reset_index()
```

---

## 3. 이상치(Outliers) 처리

### 왜 문제인가

선형 회귀(Linear Regression)의 계수(coefficient), 평균, 표준편차는 이상치 하나에도 크게 흔들립니다. 단, 트리 계열 모델(Random Forest, XGBoost)은 상대적으로 이상치에 강인합니다.

### IQR(사분위수 범위, Interquartile Range) 방법

분포가 비대칭(skewed)일 때도 잘 작동하는 방법입니다.

$$\text{lower} = Q_1 - 1.5 \times IQR, \quad \text{upper} = Q_3 + 1.5 \times IQR$$

```python
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(df['col'] < lower) | (df['col'] > upper)]
```

### z-score 방법

정규분포를 가정합니다. $|z| > 3$ 이면 이상치로 판단합니다.

$$z = \frac{x - \mu}{\sigma}$$

```python
from scipy import stats
z_scores = np.abs(stats.zscore(df['col']))
outliers = df[z_scores > 3]
```

### 처리 전략: 제거 vs 캐핑 vs 변환

**제거(Removal)** — 입력 오류가 확실한 경우에만 사용. 실제 발생 가능한 극단값을 지우면 모델이 그 구간을 학습하지 못합니다.

**캐핑(Capping / Winsorizing)** — 상·하한 경계값으로 클리핑(clipping).

```python
df['col'] = df['col'].clip(lower=lower_bound, upper=upper_bound)
```

**변환(Transformation)** — 로그 변환으로 오른쪽 꼬리가 긴 분포를 압축. 소득, 가격, 거래량 변수에 자주 사용합니다.

```python
df['col_log'] = np.log1p(df['col'])  # log(1+x): 0값 포함 안전
```

---

## 공통 핵심 원칙

| 항목 | 핵심 판단 기준 |
|---|---|
| 결측값 | 비율 + 변수 유형 + 모델과의 관계 |
| 중복값 | 완전 중복은 제거, 부분 중복은 도메인 판단 |
| 이상치 | 제거 전 반드시 "진짜 이상치인가?" 확인 |

전처리는 기계적으로 적용하는 것이 아니라, **데이터와 도메인을 동시에 이해하는 사람만이 올바르게 결정할 수 있습니다.**
