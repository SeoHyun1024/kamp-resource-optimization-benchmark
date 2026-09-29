# KAMP Resource Optimization

KAMP 자원 최적화 데이터셋(`data/okm_augumented_2021.csv`, 2021-01-01 ~ 2021-09-14, 1시간 단위 6,168행)으로 제조공정의 최대수요전력을 분석한다.

- 가이드북 부록의 데이터 품질지표(6대 지표)를 재현하고, 가이드북 모델(RNN, Random Forest)로 전력값을 **회귀** 예측한 뒤 LP로 인력배치를 최적화한다.
- 이를 확장해, 다음 시점의 **전력 피크 위험 확률**을 예측하는 **분류** 모델(XGBoost, CatBoost)을 비교한다.

## 프로젝트 구조

```text
data/
└── okm_augumented_2021.csv        원본 데이터

notebooks/
├── 01_preprocessing.ipynb         데이터 품질지표 + 전처리 + SimpleRNN (15분 전력 회귀)
├── 02_random_forest.ipynb         Random Forest ('평균' 전력 회귀) + LP 인력배치 최적화
├── 03_xgboost.ipynb               피크 위험 분류: RF baseline vs XGBoost
└── 04_catboost.ipynb              피크 위험 분류: CatBoost (03과 동일 조건)

results/                           notebook 산출물 (CSV / JSON)
models/                            학습된 분류 모델
```

## 실행 순서

앞 notebook의 산출물을 뒤 notebook이 읽으므로 **번호 순서대로** 실행한다.

| 순서 | notebook | 읽는 파일 | 만드는 파일 |
| --- | --- | --- | --- |
| 1 | `01_preprocessing` | `data/okm_augumented_2021.csv` | `df_frame.csv`, `quality_metrics.txt`, `rnn_forecast_result.csv`, `rnn_metrics.txt`, `rnn_comparison.csv`, `rnn_*_plot.png` |
| 2 | `02_random_forest` | `data/okm_augumented_2021.csv` | `rf_metrics.txt`, `rf_feature_importance.csv`, `lp_optimization_result.csv`, `lp_metrics.txt` |
| 3 | `03_xgboost` | `df_frame.csv` | `peak_dataset.csv`, `peak_baseline_probabilities.csv`, `peak_experiment_meta.json`, `xgboost_predictions.csv`, `model_comparison.csv`, `models/xgboost_classifier.*` |
| 4 | `04_catboost` | 03의 산출물 | `catboost_predictions.csv`, `model_comparison.csv`(CatBoost 행 추가), `models/catboost_classifier.*` |

- 04는 03에서 저장한 데이터셋·분할·RF/XGBoost 확률을 그대로 쓰므로 RF와 XGBoost를 다시 학습하지 않는다.
- 피크 기준 `PEAK_QUANTILE`을 바꾸려면 03과 04에서 같은 값으로 고치고 03부터 다시 실행한다. 값이 다르면 04가 assert로 멈춘다.
- 모든 notebook은 seed를 고정했으므로, 다시 실행해도 같은 결과가 나온다.

## 1. 전력 회귀 (가이드북 모델)

### 데이터 품질지표 — `01_preprocessing.ipynb`

가이드북 부록 4장의 산식으로 계산한다. 무결성은 가이드북에 공식이 없어, 측정 가능한 4개 지표의 단순평균으로 근사했다(우리 정의).

| 지표 | 가이드북 참고값 | 보정 전 | 보정 후 |
| --- | --- | --- | --- |
| 완전성 | 99.68 ~ 100% | 99.98% | 99.98% |
| 유일성 | 99.69% | 100.00% | 100.00% |
| 유효성 | 100.00% | 99.22% | 100.00% |
| 일관성 | 100.00% | 100.00% | 100.00% |
| 무결성(근사) | — | — | 99.995% |

- 유효성 보정: `시간` 컬럼이 깨진 48행(2021-07-13, 07-15)을 날짜별 행 순서로 0~23시로 재구성했다. 보정 결과는 `df_frame.csv`에 반영된다.

### SimpleRNN — `01_preprocessing.ipynb`

과거 168시간(t-1 ~ t-168)의 15분 최대수요전력으로 **1시간 뒤** 값을 예측한다. test는 마지막 336시간(2021-09-01 ~ 09-14)이며, 각 시점마다 실제 관측된 과거 값을 입력으로 쓴다(1-step-ahead).
같은 lag 피처와 test 구간으로 두 가지 방식을 비교한다.

| | 5장: 가이드북 재현 | 7장: 이전 방식 보완 |
| --- | --- | --- |
| 입력 | (168 타임스텝, 1), 최신 → 과거 | (7일, 24시간), 과거 → 최신 |
| 구조 | SimpleRNN(64) → Dense(32) → Dense(1) | SimpleRNN(50) → Dense(1) |
| early stopping | train loss, patience 5 | validation(2021-08-18 ~ 08-31) loss, patience 30 |

| test (9/1 ~ 9/14) | MAE | MSE | RMSE |
| --- | --- | --- | --- |
| **SimpleRNN (7장, 이전 방식 보완)** | **7.87** | **146.05** | **12.09** |
| SimpleRNN (5장, 가이드북 재현) | 15.91 | 386.57 | 19.66 |
| 1주 전 같은 시각 값 | 9.19 | 172.70 | 13.14 |
| 1시간 전 값 | 19.99 | 844.30 | 29.06 |

- 가이드북 재현 방식은 검증 기준 없이 train loss로 학습을 멈춰 "1주 전 같은 시각" 기준선보다 나쁘다. validation으로 early stopping을 하는 보완 방식은 기준선보다 MAE가 약 14% 낮다.
- 두 모델 모두 `keras.utils.set_random_seed(42)` + op determinism으로 재실행 결과를 고정했다.

### Random Forest — `02_random_forest.ipynb`

- 타깃은 `평균`. 누수를 막기 위해 15분/30분/45분/60분과 `날짜`를 피처에서 뺐다.
- 결측치: 풍속·강수량은 보간, 공장인원은 0으로 채운다. `시간` 이상치는 01과 같은 방식으로 보정한다.
- 분할: 2021-09-01 ~ 09-14(336시간) test / 나머지 train. `RandomForestRegressor(n_estimators=100)`

| | MSE | 가이드북 참고값 |
| --- | --- | --- |
| train | 23.25 | 185.98 |
| test | 130.82 | 183.06 |

- 변수중요도는 생산량(0.38) > 공장인원(0.25) > 시간(0.11) 순이다.

### LP 인력배치 최적화 — `02_random_forest.ipynb`

가이드북이 공개한 제약조건(`1 <= P_electric <= 2`, `1 <= P_human <= 75`, `11 <= 2*P_electric + P_human <= 95`)을 `scipy.optimize.linprog`(HiGHS)로 푼다.

- **Part A (가이드북 그대로)**: 목적함수 `P_human + P_electric`. 해는 P_electric 2, P_human 7, Cost 9다. 가이드북 예시(공장직원 5.0명, 최종값 610.42)와 다르며, 가이드북이 생산량을 LP에 연결하는 방식을 공개하지 않아 동일 재현은 불가능하다.
- **Part B (우리의 확장, 참고용)**: 목적함수 계수를 각 시간의 `전기요금(계절)`, `인건비`로 바꿔 6,168개 시간마다 LP를 푼다. 모든 시간에서 P_electric 1, P_human 9가 나오며 합산 최소비용은 1,076,826.9다.

## 2. 피크 위험 분류 — `03_xgboost.ipynb`, `04_catboost.ipynb`

**문제 정의**: t 시점까지의 정보(`X_t`)로 다음 시점의 피크 여부(`y_(t+1)`)를 예측한다.

- **피크**: 다음 시점의 시간당 최대수요전력(`max(15분, 30분, 45분, 60분)`)이 train 구간 상위 10%(= 179) 이상이면 1
- **Feature (30개)**
  - 전력 lag: 1, 2, 3, 6, 12, 24, 48, 168시간 전
  - 직전 시간의 15분 단위 값, rolling 통계
  - 생산량, 기상, 전기요금, 공장인원, 인건비 (모두 1시간 전 값)
  - 달력: 시간, 요일, 월, 주말, 공휴일
- **분할**

  | 구간 | 기간 | 용도 |
  | --- | --- | --- |
  | Train | 2021-01-08 ~ 07-31 | 학습, 피크 임계값 계산 |
  | Validation | 2021-08-01 ~ 08-31 | 모델 선택, 분류 threshold 선택 |
  | Test | 2021-09-01 ~ 09-14 | 최종 평가 1회 |

- **누수 방지**
  - 피크 임계값은 train으로만 계산한다.
  - hyperparameter, class balancing, early stopping, 분류 threshold는 모두 validation으로만 고른다.
  - lag가 실제로 직전 시점 값인지 notebook 안에서 assert로 확인한다.

### 결과 (test, 양성 47 / 336건)

각 모델은 validation F1이 최대인 threshold를 그대로 test에 적용했다.

| Model | threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.50 | 0.619 | **0.830** | **0.709** | 0.955 | 0.737 |
| XGBoost | 0.65 | 0.607 | 0.787 | 0.685 | 0.953 | 0.711 |
| CatBoost | 0.80 | **0.648** | 0.745 | 0.693 | **0.959** | **0.758** |
| 직전 전력 ≥ 179 (규칙) | — | 0.447 | 0.447 | 0.447 | 0.862 | 0.530 |

- **CatBoost**
  - threshold와 무관한 순위 품질(PR-AUC, ROC-AUC)이 가장 좋다.
  - threshold가 높아(0.8) 오탐은 가장 적고(19건), 놓친 피크는 가장 많다(12건).
- **F1**: 세 모델의 차이가 작다. 양성이 47건뿐이라 피크 1건이 Recall 약 0.02에 해당한다.
- **중요 변수**
  - XGBoost는 `lag_168`(지난주 같은 시각)을 가장 많이 쓴다.
  - CatBoost는 범주형 `hour`(가동 시간대)를 가장 많이 쓴다.
  - 생산량과 기상 변수의 영향은 작다.
- CatBoost의 달력 변수는 범주형 방식과 cyclic(sin/cos) 방식을 비교했고, validation에서 범주형 방식이 더 좋았다.
- 전체 비교표(기본 모델, threshold 0.5 결과 포함)는 `results/model_comparison.csv`에 있다.

## 데이터 주의사항

- **`시간` 컬럼 오류**: 2021-07-13, 07-15 이틀치 `시간` 값이 70~188로 깨져 있다. 하루 24행은 유지되어 있어, 01에서 날짜별 행 순서로 0~23시를 다시 채워 `df_frame.csv`에 저장한다.
- **복사된 구간**: 증강 데이터라 이전 달의 구간이 다른 날짜에 그대로 복사되어 있다. 한 시간의 15분 값 4개 조합이 이전 달에도 존재하는 비율은 6월 96%, 8월 27%, 9월 19%다.
  - 1~7월을 검증에 쓰면 모델이 복사본을 외운 것까지 성능으로 잡힌다.
  - 모델 평가는 9월 test 구간을 기준으로 본다.
  - 8월 validation 점수가 test보다 좋게 나오는 원인 중 하나다.
- **9월은 train에 없는 월**: train은 1~7월이라 test의 `month`(9월) 값을 학습 중에 본 적이 없다. 그래서 month 변수는 test 예측에 거의 기여하지 않는다.

## 실행 방법

### 1. 가상환경 생성

```bash
python -m venv .venv
```

### 2. 가상환경 활성화

Mac / Linux

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

macOS에서 XGBoost를 사용하려면 OpenMP 런타임이 필요하다.

```bash
brew install libomp
```

### 4. VS Code에서 실행

1. VS Code에서 프로젝트 폴더 열기
2. Python, Jupyter Extension 설치
3. `.ipynb` 파일 열기
4. 우측 상단 `Select Kernel` 클릭
5. `.venv` 환경 선택
6. `01` → `04` 순서로, 각 notebook의 셀을 순서대로 실행
