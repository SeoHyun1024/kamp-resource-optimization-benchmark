# KAMP Resource Optimization

KAMP 자원 최적화 데이터셋(`data/okm_augumented_2021.csv`, 2021-01-01 ~ 2021-09-14, 1시간 단위 6,168행)으로 제조공정의 최대수요전력을 분석한다.

- 가이드북 모델(RNN, Random Forest)로 전력값을 **회귀** 예측한다.
- 이를 확장해, 다음 시점의 **전력 피크 위험 확률**을 예측하는 **분류** 모델(XGBoost, CatBoost)을 비교한다.

## 폴더 구조

```text
data/
└── okm_augumented_2021.csv        원본 데이터

notebooks/
├── 01_preprocessing.ipynb         전처리 + SimpleRNN (1시간 앞 전력 회귀)
├── 02_random_forest.ipynb         Random Forest (다음 15분 전력 회귀) + OR-Tools 인력 최적화
├── 03_xgboost.ipynb               피크 위험 분류: RF baseline vs XGBoost
└── 04_catboost.ipynb              피크 위험 분류: CatBoost (03과 동일 조건)

results/                           notebook 산출물 (CSV / JSON)
models/                            학습된 분류 모델
```

## 실행 순서

앞 notebook의 산출물을 뒤 notebook이 읽으므로 **번호 순서대로** 실행한다.

| 순서 | notebook           | 읽는 파일                      | 만드는 파일                                                                                                                                                          |
| ---- | ------------------ | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `01_preprocessing` | `data/okm_augumented_2021.csv` | `df_frame.csv`, `daily_data_seasonality*.csv`, `the_best_rnn_pred_168.csv`                                                                                           |
| 2    | `02_random_forest` | `df_frame.csv`                 | —                                                                                                                                                                    |
| 3    | `03_xgboost`       | `df_frame.csv`                 | `peak_dataset.csv`, `peak_baseline_probabilities.csv`, `peak_experiment_meta.json`, `xgboost_predictions.csv`, `model_comparison.csv`, `models/xgboost_classifier.*` |
| 4    | `04_catboost`      | 03의 산출물                    | `catboost_predictions.csv`, `model_comparison.csv`(CatBoost 행 추가), `models/catboost_classifier.*`                                                                 |

- 04는 03에서 저장한 데이터셋·분할·RF/XGBoost 확률을 그대로 쓰므로 RF와 XGBoost를 다시 학습하지 않는다.
- 피크 기준 `PEAK_QUANTILE`을 바꾸려면 03과 04에서 같은 값으로 고치고 03부터 다시 실행한다. 값이 다르면 04가 assert로 멈춘다.
- 모든 notebook은 seed를 고정했으므로, 다시 실행해도 같은 결과가 나온다.

## 1. 전력 회귀 (가이드북 모델)

### SimpleRNN — `01_preprocessing.ipynb`

- 과거 168시간(7일 × 24시간)의 15분 최대수요전력으로 **1시간 뒤** 값을 예측한다.
- test 구간(2021-09-01 ~ 09-14)의 각 시점마다, 실제로 관측된 과거 값을 입력으로 쓴다(1-step-ahead). 168시간을 한 번에 예측하는 것이 아니다.
- 분할: train ~ 2021-08-17 / validation 2021-08-18 ~ 08-31 (early stopping) / test 2021-09-01 ~ 09-14

| test (9/1 ~ 9/14)   | MAE      | RMSE      |
| ------------------- | -------- | --------- |
| **SimpleRNN**       | **7.87** | **12.09** |
| 1주 전 같은 시각 값 | 9.19     | 13.14     |
| 1시간 전 값         | 19.99    | 29.06     |

### Random Forest — `02_random_forest.ipynb`

- 현재 15분 전력과 시간·기상·달력 변수로 **다음 15분** 전력을 예측한다. 분할은 시간순으로 70% / 30%다.
- test MAE는 8.58로, **직전 값을 그대로 쓰는 기준선(7.93)보다 나쁘다.** 입력 정보가 부족해 구조적으로 한계가 있다.
- 예측값은 OR-Tools 선형계획(주/야간, 투입 인원)의 입력으로 쓰인다.

## 2. 피크 위험 분류 — `03_xgboost.ipynb`, `04_catboost.ipynb`

**문제 정의**: t 시점까지의 정보(`X_t`)로 다음 시점의 피크 여부(`y_(t+1)`)를 예측한다.

- **피크**: 다음 시점의 시간당 최대수요전력(`max(15분, 30분, 45분, 60분)`)이 train 구간 상위 10%(= 179) 이상이면 1
- **Feature (30개)**
  - 전력 lag: 1, 2, 3, 6, 12, 24, 48, 168시간 전
  - 직전 시간의 15분 단위 값, rolling 통계
  - 생산량, 기상, 전기요금, 공장인원, 인건비 (모두 1시간 전 값)
  - 달력: 시간, 요일, 월, 주말, 공휴일
- **분할**

  | 구간       | 기간               | 용도                           |
  | ---------- | ------------------ | ------------------------------ |
  | Train      | 2021-01-08 ~ 07-31 | 학습, 피크 임계값 계산         |
  | Validation | 2021-08-01 ~ 08-31 | 모델 선택, 분류 threshold 선택 |
  | Test       | 2021-09-01 ~ 09-14 | 최종 평가 1회                  |

- **누수 방지**
  - 피크 임계값은 train으로만 계산한다.
  - hyperparameter, class balancing, early stopping, 분류 threshold는 모두 validation으로만 고른다.
  - lag가 실제로 직전 시점 값인지 notebook 안에서 assert로 확인한다.

### 결과 (test, 양성 47 / 336건)

각 모델은 validation F1이 최대인 threshold를 그대로 test에 적용했다.

| Model                  | threshold | Precision | Recall    | F1        | ROC-AUC   | PR-AUC    |
| ---------------------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Random Forest          | 0.50      | 0.619     | **0.830** | **0.709** | 0.955     | 0.737     |
| XGBoost                | 0.65      | 0.607     | 0.787     | 0.685     | 0.953     | 0.711     |
| CatBoost               | 0.80      | **0.648** | 0.745     | 0.693     | **0.959** | **0.758** |
| 직전 전력 ≥ 179 (규칙) | —         | 0.447     | 0.447     | 0.447     | 0.862     | 0.530     |

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

- **`시간` 컬럼 오류**: 2021-07-13, 07-15 이틀치 `시간` 값이 70~188로 깨져 있다. 하루 24행은 유지되어 있어, 시각은 날짜와 하루 안의 행 순서로 만든다.
- **복사된 구간**: 증강 데이터라 이전 달의 구간이 다른 날짜에 그대로 복사되어 있다. 한 시간의 15분 값 4개 조합이 이전 달에도 존재하는 비율은 6월 96%, 8월 27%, 9월 19%다.
  - 1~7월을 검증에 쓰면 모델이 복사본을 외운 것까지 성능으로 잡힌다.
  - 모델 평가는 9월 test 구간을 기준으로 본다.
  - 8월 validation 점수가 test보다 좋게 나오는 원인 중 하나다.
- **9월은 train에 없는 월**: train은 1~7월이라 test의 `month`(9월) 값을 학습 중에 본 적이 없다. 그래서 month 변수는 test 예측에 거의 기여하지 않는다.

## 실행 방법

Python 3.11 기준입니다.

```bash
pip install -r requirements.txt

python preprocessing.py        # 전처리 → results/1_preprocessing/
python 01_rnn.py               # RNN (CV 포함 약 1시간, 빠른 실행: --variant D_multi_gru --epochs 90)
python 02_random_forest.py
python 03_xgboost.py
python 04_catboost.py
python 05_isolation_forest.py
python 06_regression.py
python 07_compare.py           # 마지막에 실행 → results/3_comparison/final_comparison.csv
python ablation_missing.py     # (선택) 결측 처리 방식 비교, 약 4분
```

- 모든 스크립트는 `kamp` 폴더에서 실행합니다.
- 시드는 42로 고정했습니다. 다만 CPU나 라이브러리 버전에 따라 소수점 수준의 차이는 날 수 있습니다.

## 전처리

원자료를 진단한 결과와 처리 방법입니다. 실행 로그는 `results/1_preprocessing/preprocessing_report.txt`에 남습니다.

| 항목          | 진단 결과                                                         | 처리                                                                                                                                                                                             |
| ------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 결측치        | 풍속 3건, 강수량 1건, 공장인원 17건                               | 풍속·강수량은 시간 보간. 공장인원은 결측 행이 전부 생산량 0 구간임을 확인한 뒤 0으로 채움. 보간·직전값·0·모델 내장 처리 4가지를 비교했는데 성능 차이가 없어 보간을 유지함(`ablation_missing.py`) |
| `시간` 이상치 | 2021-07-13, 07-15 두 날의 48행이 0~23 범위를 벗어남(70~188)       | 두 날 모두 24행이 온전하므로 날짜 안의 행 순서로 0~23시를 재구성                                                                                                                                 |
| 증강 복제일   | 257일 중 115일이 앞선 날의 15분 값 96개와 완전히 같음. 모두 1~7월 | 행은 유지하고 `is_copied_day` 플래그만 만듦. CV 검증 구간은 복제가 없는 7·8월로 잡음. 학습에서 빼면 CV 성능이 떨어져(`ablation_copied_days.py`) 학습에는 포함                                    |
| 공휴일        | 1/1, 설, 3/1, 5/5, 5/19, 8/16이 모두 가동일(일 최대전력 108~195)  | 공휴일 변수를 쓰지 않음                                                                                                                                                                          |
| 주말          | 토요일 37일 중 35일 가동                                          | 주말 변수 대신 `day_of_week` 사용                                                                                                                                                                |
| `month`       | test(9월)가 학습 데이터에 없는 값                                 | 사용하지 않음                                                                                                                                                                                    |

**예측 대상**

- `power` = max(15분, 30분, 45분, 60분): 해당 시간의 최대수요전력입니다.
- 피크 라벨은 `power ≥ 179`입니다. 179는 2021-01-08~07-31 `power`의 90% 분위수예요.

## Feature (38개, 모두 T−1 시점까지의 정보)

| 그룹                    | Feature                                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| 전력 lag                | `lag_1, 2, 3, 6, 12, 24, 48, 167, 168, 169, 336`                                            |
| 직전 1시간 15분 단위 값 | `q15_lag1`, `q30_lag1`, `q45_lag1`, `q60_lag1`, `avg_lag1`, `intra_hour_slope`(60분 − 15분) |
| 추세·통계               | `diff_1`, `roll_mean_3`, `roll_mean_24`, `roll_max_24`, `roll_std_24`, `max_since_midnight` |
| 주간 패턴               | `same_hour_2w_mean`, `same_hour_2w_max`, `last_op_day_same_hour`(직전 가동일 같은 시각)     |
| 전일 상태               | `prev_day_max`, `prev_day_off`                                                              |
| 외생변수(1시간 전)      | 생산량, 기온, 풍속, 습도, 강수량, 전기요금, 공장인원, 인건비                                |
| 달력                    | `hour`, `day_of_week`                                                                       |

**누수 점검**: `preprocessing.leakage_checks()`에서 다음을 assert로 확인합니다.

- `lag_1`, `q60_lag1`, `production`이 T−1 값인지
- target과 같거나 상관계수가 0.99 이상인 feature가 없는지

## 검증 방식

- **CV**: 확장창 2-fold로 hyperparameter, threshold, 모델 구조를 선택합니다.
  - fold 1: 7/1 이전으로 학습하고 7월로 검증
  - fold 2: 8/1 이전으로 학습하고 8월로 검증
- **최종 학습**: 9/1 이전 전체 데이터로 다시 학습합니다.
- **Test**: 2021-09-01~09-14(336시간, 피크 47건)이며, 선택에는 쓰지 않고 마지막에 1회만 평가합니다.

## 결과 (Test 2021-09-01~09-14)

| 모델                                   | CV F1 | CV PR-AUC | Test F1 (95% CI)   | Test PR-AUC | Test Recall |
| -------------------------------------- | ----- | --------- | ------------------ | ----------- | ----------- |
| 규칙: 직전 값 (lag_1 ≥ 179)            | –     | –         | 0.447              | 0.530       | 0.447       |
| 규칙: 1주 전 같은 시각 (lag_168 ≥ 179) | –     | –         | 0.654              | 0.562       | 0.702       |
| Random Forest                          | 0.817 | 0.886     | 0.703 [0.53, 0.82] | 0.767       | 0.830       |
| XGBoost                                | 0.820 | 0.895     | 0.691 [0.55, 0.80] | 0.740       | 0.809       |
| CatBoost                               | 0.827 | 0.905     | 0.679 [0.54, 0.78] | 0.738       | 0.809       |
| 앙상블 (RF+XGB+CB 평균)                | 0.829 | 0.905     | 0.685 [0.52, 0.80] | 0.748       | 0.787       |
| RNN (GRU, 예측값 ≥ 167)                | –     | –         | 0.732 [0.59, 0.83] | 0.744       | 0.957       |
| XGBoost 회귀 (예측값 ≥ 171)            | 0.794 | –         | 0.694 [0.54, 0.81] | 0.697       | 0.894       |
| LightGBM 회귀 (예측값 ≥ 173)           | 0.806 | –         | 0.690 [0.52, 0.80] | 0.685       | 0.851       |
| Isolation Forest (비지도)              | 0.618 | 0.504     | 0.485 [0.37, 0.58] | 0.344       | 0.702       |

회귀 성능(Test MAE):

| 모델                       | MAE   | RMSE  |
| -------------------------- | ----- | ----- |
| 직전 값 (lag_1)            | 14.52 | 23.78 |
| 1주 전 같은 시각 (lag_168) | 8.95  | 12.33 |
| XGBoost 회귀               | 6.22  | 8.88  |
| LightGBM 회귀              | 6.27  | 9.79  |
| RNN (GRU)                  | 6.72  | 9.13  |

**RNN 구조 선택 결과**: CV MAE 기준으로 GRU가 선택됐습니다.

| 구조         | 입력 형태 | CV MAE   |
| ------------ | --------- | -------- |
| A: SimpleRNN | (168, 1)  | 11.39    |
| B: SimpleRNN | (7, 24)   | 14.14    |
| C: SimpleRNN | (168, 8)  | 9.59     |
| D: GRU       | (168, 8)  | **8.24** |

### 해석

1. **지도학습 모델은 모두 규칙 기반 기준선보다 좋습니다.** RF, XGBoost, CatBoost, RNN, 회귀 모델의 F1은 0.68~0.73입니다.
   - 규칙 기반 기준선은 직전 값 0.447, 1주 전 같은 시각 0.654입니다.
2. **모델 간 차이는 통계적으로 확정할 수 없습니다.**
   - test가 14일(피크 47건)뿐이라 F1 95% 신뢰구간 폭이 약 ±0.15입니다.
   - 신뢰구간은 일 단위 블록 부트스트랩(2000회)으로 구했습니다.
   - 그래서 모델을 고를 때는 CV 지표와 함께 판단합니다.
3. **RNN(GRU)은 Recall이 가장 높습니다(0.957).** 대신 Precision이 0.592로 오탐이 많습니다.
   - 피크를 놓치지 않는 것이 중요한 경보 용도에 적합합니다.
4. **Isolation Forest는 라벨 없이 동작하지만 성능이 낮습니다.**
   - F1 0.485로, 1주 전 같은 시각 규칙보다도 낮습니다.
   - 라벨이 없는 신규 설비용 참고 모델로만 둡니다.
5. **중요 변수**: 1·2주 전 같은 시각 전력(`same_hour_2w_max`), 직전 시간 마지막 15분 값(`q60_lag1`), 시간대(`hour`)가 모든 모델에서 상위권입니다.
   - 즉 공장의 **주간 가동 패턴**과 **직전 추세**가 피크를 가장 잘 설명합니다.

## 결과 파일 (`results/`)

| 폴더                  | 파일                                                                                        | 내용                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `1_preprocessing/`    | `clean_hourly.csv`                                                                          | 정제된 시간 단위 데이터                                               |
|                       | `peak_dataset.csv`                                                                          | 모델 공통 feature 38개 + target + 라벨                                |
|                       | `copied_days.json`                                                                          | 복제일 → 원본 날짜                                                    |
|                       | `preprocessing_report.txt`                                                                  | 전처리 진단 로그                                                      |
| `2_test_predictions/` | `rnn_forecast.csv`                                                                          | RNN: `Date, actual, forecast, actual_label, predicted_label`          |
|                       | `random_forest_predictions.csv`                                                             | `Date, actual_label, predicted_label, risk_probability, actual_power` |
|                       | `xgboost_predictions.csv`, `catboost_predictions.csv`, `ensemble_rf_xgb_cb_predictions.csv` | 같은 형식                                                             |
|                       | `xgboost_reg_predictions.csv`, `lightgbm_reg_predictions.csv`                               | 회귀: `predicted_power` 포함                                          |
|                       | `isolation_forest_predictions.csv`                                                          | `anomaly_score` 포함                                                  |
|                       | `persistence_predictions.csv`, `persistence_lag168_predictions.csv`                         | 규칙 기반 기준선                                                      |
| `3_comparison/`       | `final_comparison.csv`                                                                      | **전체 모델 test 성능 + F1 95% 신뢰구간 (보고서용)**                  |
|                       | `model_comparison.csv`                                                                      | 모델별 test 성능                                                      |
|                       | `regression_comparison.csv`                                                                 | 회귀 MAE/RMSE                                                         |
|                       | `rnn_metrics.txt`                                                                           | RNN test 성능 요약                                                    |
|                       | `missing_ablation_summary.csv`                                                              | 결측 처리 방식별 성능 비교 요약 (`ablation_missing.py`)               |
| `4_model_details/`    | `*_cv.csv`, `*_oof.csv`, `*_importance.csv`                                                 | grid 탐색 결과, CV out-of-fold 예측, 변수 중요도                      |
|                       | `rnn_cv.csv`                                                                                | RNN 구조별 CV 결과 (CV 포함 실행 시)                                  |
|                       | `missing_ablation_real.csv`, `missing_ablation_simulated.csv`                               | 결측 처리 비교 상세                                                   |
| `5_models/`           | `xgboost_classifier.json`, `catboost_classifier.cbm`                                        | 학습된 모델                                                           |
| `6_figures/`          | `rnn_forecast_plot.png`                                                                     | RNN 테스트 예측 그래프                                                |
