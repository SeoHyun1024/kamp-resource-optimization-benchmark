# Hyeonjun 재현 실행 결과

README.md의 실행 방법을 `.venv`로 처음부터 끝까지 다시 실행한 기록입니다. 코드는 수정하지 않았습니다.

## 1. 실행 환경

| 항목 | 값 |
| --- | --- |
| OS | Windows 11 Pro |
| Python | 3.11.9 (`.venv`) |
| numpy / pandas | 2.4.6 / 3.0.6 |
| scikit-learn | 1.9.1 |
| xgboost / lightgbm / catboost | 3.2.0 / 4.7.0 / 1.2.10 |
| tensorflow | 2.21.0 |
| 시드 | 42 (코드 고정) |

## 2. 실행 내역

모든 스크립트가 종료 코드 0으로 끝났습니다. `01_rnn.py`는 단독 실행 파일이라 `preprocessing.py` 계열과 병렬로 돌렸습니다.

| 순서 | 명령 | 소요 시간 | README 예상 |
| --- | --- | --- | --- |
| 1 | `python preprocessing.py` | 3초 | – |
| 2 | `python 01_rnn.py --variant D_multi_gru --epochs 90` | 280초 | 10~15분 |
| 3 | `python 02_random_forest.py` | 64초 | 약 2분 |
| 4 | `python 03_xgboost.py` | 19초 | 약 30초 |
| 5 | `python 04_catboost.py` | 228초 | 약 1분 |
| 6 | `python 05_isolation_forest.py` | 6초 | 약 10초 |
| 7 | `python 06_regression.py` | 38초 | 약 1~2분 |
| 8 | `python 07_compare.py` | 28초 | – |
| 9 | `python ablation_copied_days.py` | 7초 | 약 1분 |
| 10 | `python ablation_missing.py` | 446초 | 약 4분 |
| 11 | `python ablation_encoding.py` | 34초 | 약 1분 |

`04_catboost.py`와 `ablation_missing.py`는 README 예상보다 오래 걸렸습니다. 이 PC에서는 CatBoost가 약 4배, 결측 ablation이 약 2배 걸렸습니다.

## 3. 재현성 점검

저장소에 커밋된 결과와 비교했습니다.

- `results/3_comparison/final_comparison.csv`와 `regression_comparison.csv`는 git diff가 없습니다. 지표가 완전히 일치합니다.
- 바뀐 파일은 5개입니다. `random_forest_predictions.csv`, `ensemble_rf_xgb_cb_predictions.csv`, `random_forest_oof.csv`는 확률값의 마지막 유효숫자(예: `…0004099151768` → `…0004099151767`)만 다릅니다. `catboost_classifier.cbm`, `rnn.keras`는 바이너리 재생성에 따른 변경이며 크기는 같습니다.
- 결론: 시드 고정 덕분에 지표 수준에서 재현됐습니다.

## 4. Test 결과 (2021-09-01~09-14, 336시간, 피크 47건)

`results/3_comparison/final_comparison.csv` 기준입니다.

| 모델 | threshold | Precision | Recall | F1 | F1 95% CI | PR-AUC | CV F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 규칙: 직전 값 | 179 | 0.447 | 0.447 | 0.447 | [0.195, 0.605] | 0.529 | – |
| 규칙: 1주 전 같은 시각 | 179 | 0.611 | 0.702 | 0.653 | [0.515, 0.768] | 0.561 | – |
| Random Forest | 0.55 | 0.606 | 0.915 | **0.729** | [0.584, 0.830] | **0.767** | 0.821 |
| XGBoost | 0.30 | 0.613 | 0.809 | 0.697 | [0.569, 0.789] | 0.723 | 0.821 |
| CatBoost | 0.30 | 0.585 | 0.809 | 0.679 | [0.536, 0.783] | 0.738 | 0.827 |
| 앙상블 (RF+XGB+CB) | 0.45 | 0.617 | 0.787 | 0.692 | [0.529, 0.804] | 0.742 | 0.829 |
| RNN (GRU) | 167 | 0.592 | **0.957** | **0.732** | [0.587, 0.832] | 0.744 | – |
| XGBoost 회귀 | 171 | 0.573 | 0.915 | 0.705 | [0.542, 0.818] | 0.728 | 0.791 |
| LightGBM 회귀 | 173 | 0.580 | 0.851 | 0.690 | [0.519, 0.802] | 0.685 | 0.806 |
| Isolation Forest | – | 0.371 | 0.702 | 0.485 | [0.370, 0.576] | 0.344 | 0.618 |

### 회귀 성능 (Test)

| 모델 | MAE | RMSE | 피크 시간대 MAE |
| --- | --- | --- | --- |
| 직전 값 (lag_1) | 14.52 | 23.78 | 22.06 |
| 1주 전 같은 시각 (lag_168) | 8.95 | 12.33 | 9.89 |
| RNN (GRU) | 6.72 | 9.13 | – |
| XGBoost 회귀 | **6.12** | 8.62 | 8.32 |
| LightGBM 회귀 | 6.28 | 9.79 | 8.40 |

## 5. 보조 실험

### 5.1 복제일 처리 (`ablation_copied_days.py`)

복제일 115일 중 83일은 원본과 요일이 다릅니다.

| 방식 | CV PR-AUC | CV F1 |
| --- | --- | --- |
| keep (전부 사용) | **0.8631** | **0.7961** |
| weight 0.5 | 0.8612 | 0.7926 |
| 요일 불일치만 제거 | 0.8498 | 0.7790 |
| 전부 제거 | 0.8368 | 0.7669 |

복제일을 학습에서 빼면 CV 성능이 떨어져서, 모두 유지하는 방식이 맞다고 확인됩니다.

### 5.2 결측 처리 (`ablation_missing.py`)

보간, 직전값, 0, 모델 내장 처리 4가지, 실제 결측/무작위 10%/블록 10% 3가지 조건으로 RF·XGB·CatBoost를 비교했습니다.

- CV PR-AUC 범위는 RF 0.882~0.888, XGB 0.893~0.895, CatBoost 0.897~0.903으로 처리 방식 간 차이가 거의 없습니다.
- Test F1은 0.66~0.71 범위에서 방식과 무관하게 흔들립니다.
- 따라서 결측 처리 방식은 결과에 의미 있는 영향을 주지 않으므로 기본값(보간)이 적절합니다.

### 5.3 요일·시간 인코딩 (`ablation_encoding.py`)

| 모델 | 인코딩 | feature 수 | CV PR-AUC | CV F1 | Test F1 |
| --- | --- | --- | --- | --- | --- |
| RF | int | 38 | 0.882 | 0.815 | 0.703 |
| RF | onehot | 67 | 0.879 | 0.806 | 0.713 |
| RF | cyclic | 40 | 0.886 | 0.814 | 0.700 |
| RF | int+cyclic | 42 | 0.889 | 0.807 | 0.722 |
| XGB | int | 38 | 0.890 | 0.811 | 0.673 |
| XGB | onehot | 67 | 0.876 | 0.808 | 0.679 |
| XGB | cyclic | 40 | 0.893 | 0.818 | 0.711 |
| XGB | int+cyclic | 42 | 0.892 | 0.819 | 0.696 |

인코딩 간 차이는 CV 표준편차(0.001~0.006) 수준이고 방향도 일관되지 않아 뚜렷한 우열이 없습니다.

### 5.4 변수 중요도

- XGBoost gain: `same_hour_2w_max` 0.249, `roll_max_24` 0.097, `q60_lag1` 0.090, `lag_168` 0.056, `hour` 0.047
- CatBoost SHAP(평균 절댓값): `hour` 0.74, `same_hour_2w_max` 0.43, `q60_lag1` 0.29, `max_since_midnight` 0.23, `roll_max_24` 0.18

모델마다 순위는 다르지만 `same_hour_2w_max`, `q60_lag1`, `hour`가 공통으로 상위입니다.

## 6. README.md와의 불일치

재실행 결과(= 커밋된 CSV)와 README.md 표가 다릅니다. README가 갱신되지 않은 것으로 보입니다.

| 항목 | README.md | 실제 결과 |
| --- | --- | --- |
| Random Forest F1 / Recall | 0.703 / 0.830 | 0.729 / 0.915 |
| XGBoost F1 | 0.691 | 0.697 |
| 앙상블 F1 | 0.685 | 0.692 |
| XGBoost 회귀 F1 | 0.694 | 0.705 |
| XGBoost 회귀 MAE | 6.22 | 6.12 |

README의 RF 0.703은 인코딩 ablation의 `int` 조건 Test F1(0.703)과 일치합니다. 최종 RF가 `int+cyclic`으로 바뀌었는데 README에는 이전 값이 남은 것으로 추정됩니다.

## 7. 해석

1. 지도학습 모델은 모두 규칙 기준선보다 좋지만(F1 0.68~0.73 대 0.45~0.65), 모델 간 차이는 F1 신뢰구간(폭 약 0.25)에 비해 작아 순위를 확정할 수 없습니다.
2. RNN(GRU)은 Recall 0.957로 가장 높고, 대신 Precision이 낮아 오탐이 31건입니다. 피크를 놓치지 않는 경보용에 맞습니다.
3. Random Forest는 분류 모델 중 Test F1과 PR-AUC가 가장 높습니다. 다만 CV에서는 CatBoost와 앙상블이 더 높아, 선택 기준에 따라 순위가 달라집니다.
4. 회귀 모델은 MAE 6.1~6.3으로 가장 정확하고, 예측값을 threshold로 잘라 피크 판정에도 쓸 수 있습니다.
5. Isolation Forest는 지도 모델보다 한참 낮아 참고용입니다.
6. Test가 14일이라 변동이 크므로, 보고 시 단일 점수보다 신뢰구간과 CV 지표를 함께 제시해야 합니다.
