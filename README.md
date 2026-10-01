# 피크전력 위험 예측 모델

제조 공장의 시간당 최대수요전력을 이용해 **다음 1시간이 피크(≥179)가 될 위험**을 예측합니다. 사용한 모델은 RNN, Random Forest, XGBoost, CatBoost, Isolation Forest, 회귀(XGBoost·LightGBM) 입니다.

## 폴더 구조

```
kamp/
├─ data/
│  └─ okm_augumented_2021.csv   # 학습용 데이터 (원자료)
├─ preprocessing.py             # 전처리 + feature 생성 (02~06이 import)
├─ common.py                    # CV 분할·평가·threshold 선택·결과 저장 공통 함수
├─ 01_rnn.py                    # RNN (단독 실행 파일, 전처리 포함)
├─ 02_random_forest.py          # Random Forest + 규칙 기반 기준선 2종
├─ 03_xgboost.py
├─ 04_catboost.py
├─ 05_isolation_forest.py
├─ 06_regression.py             # XGBoost / LightGBM 회귀
├─ 07_compare.py                # 앙상블 + 전체 비교표 + 부트스트랩 신뢰구간
├─ ablation_copied_days.py      # (선택) 복제일 처리 방식 비교 실험
├─ ablation_missing.py          # (선택) 결측 처리 방식별 성능 비교 실험
├─ requirements.txt
└─ results/                     # 실행하면 자동 생성
   ├─ 1_preprocessing/          # 전처리 결과
   ├─ 2_test_predictions/       # 테스트데이터 예측결과 (제출용)
   ├─ 3_comparison/             # 모델 성능 비교표
   ├─ 4_model_details/          # CV 탐색표·OOF 예측·변수 중요도
   ├─ 5_models/                 # 학습된 모델 파일
   └─ 6_figures/                # 그림
```

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
