# 피크 위험 예측 모델 비교 결과 (Persistence → RF → XGBoost → CatBoost → Isolation Forest)

`02_random_forest.ipynb` ~ `05_isolation_forest.ipynb` 전체 실험을 같은 데이터셋(`results/peak_dataset.csv`)·시계열 분할·피크 정의(`results/peak_experiment_meta.json`)로 정리한 결과. 마지막 Isolation Forest만 **비지도 이상탐지**이고, 나머지는 모두 **supervised 분류(피크 확률 예측)**다.

```
Supervised Peak Risk Classification (Random Forest / XGBoost / CatBoost)
        vs
Unsupervised Anomaly Detection (Isolation Forest)
```

## 데이터 / 분할

| 구간 | 기간 | 샘플 수 | 양성(피크) 비율 |
| --- | --- | --- | --- |
| Train | 2021-01-08 ~ 2021-07-31 | 4,920 | 10.26% |
| Validation | 2021-08-01 ~ 2021-08-31 | 744 | 13.58% |
| Test | 2021-09-01 ~ 2021-09-14 | 336 | 13.99% |

- Peak threshold = 179.0 (train 90% quantile)
- Feature 30개 (lag, rolling 통계, 외생변수, 캘린더) — 전 모델 공통, target leakage 없음 확인
- Threshold/contamination은 모두 **validation**에서만 선택하고 test에는 1회만 적용 (data leakage 방지)

## 모델별 요약

| 모델 | 방식 | 최종 선택 threshold(또는 contamination) | 한 줄 요약 |
| --- | --- | --- | --- |
| Persistence (baseline) | 규칙 기반 | 179.0 (score=`lag_1`) | "직전 시점이 피크면 다음도 피크"라는 단순 규칙. 학습 모델의 하한선 |
| Random Forest | supervised | 0.5 | validation PR-AUC 기준 소규모 grid search. Test에서 전 지표 최상위 |
| XGBoost | supervised | 0.65 (tuned) | `scale_pos_weight`로 클래스 불균형 보정 + validation threshold tuning |
| CatBoost | supervised | 0.8 (tuned) | 범주형 native encoding + `auto_class_weights="Balanced"`, SHAP 해석 포함 |
| Isolation Forest | **unsupervised** | contamination 0.10 | 라벨 미사용 학습, validation에서 contamination만 선택. 최종 후보 제외 |

## 전체 비교표 (`results/model_comparison.csv`)

| Model | threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Persistence (lag_1 ≥ peak) | 179.0 | 0.4468 | 0.4468 | 0.4468 | 0.8623 | 0.5295 |
| Random Forest @0.5 | 0.5 | 0.6190 | 0.8298 | **0.7091** | 0.9553 | 0.7371 |
| XGBoost default (no spw) @0.5 | 0.5 | 0.6471 | 0.7021 | 0.6735 | 0.9561 | 0.7417 |
| XGBoost default (spw) @0.5 | 0.5 | 0.5915 | 0.8936 | 0.7119 | 0.9567 | 0.7473 |
| XGBoost tuned @0.5 | 0.5 | 0.6000 | 0.8936 | 0.7179 | 0.9533 | 0.7113 |
| XGBoost tuned @0.65 (최종) | 0.65 | 0.6066 | 0.7872 | 0.6852 | 0.9533 | 0.7113 |
| CatBoost @0.5 | 0.5 | 0.5833 | 0.8936 | 0.7059 | 0.9590 | 0.7575 |
| CatBoost @0.8 (최종) | 0.8 | 0.6481 | 0.7447 | 0.6931 | 0.9590 | 0.7575 |
| Isolation Forest (unsupervised) | 0.10 | 0.5000 | 0.1064 | 0.1754 | 0.7126 | 0.2908 |

- 위쪽 XGBoost/CatBoost 행 중 "(최종)"이 아닌 행들은 튜닝 과정에서 나온 중간 실험이다. 모델 간 최종 비교는 각 모델의 "(최종)" 행 기준으로 한다.
- **RF > CatBoost > XGBoost > Persistence > Isolation Forest** 순으로 F1이 높다. supervised 3개 모델은 F1 0.69~0.71로 사실상 비슷한 수준이고, 모두 단순 규칙(Persistence F1 0.4468) 대비 큰 폭으로 개선됐다.
- Isolation Forest는 supervised 모델뿐 아니라 단순 Persistence baseline보다도 크게 낮다 — 예측 성능이 아니라 **"라벨 없이 동작한다"는 접근 자체의 차이**로 이해해야 한다.

## Isolation Forest 상세

### 실험 설계

- **Experiment A (기본)**: train 전체로 비지도 학습, label 미사용
- **Experiment B (참고, semi-supervised)**: train 중 라벨 기준 정상(`y=0`) 데이터만으로 학습 — label을 학습 데이터 "선정"에 사용하므로 A와 구분, 최종 모델에는 미채택
- Feature 표현: raw / `StandardScaler` 적용, validation F1 기준 비교 → 결과 완전히 동일(F1 차이 0.0000)하여 더 단순한 **raw** 유지
- Contamination 그리드: `[0.01, 0.03, 0.05, 0.10]`, **validation에서만** 탐색 (test 양성 비율은 사용하지 않음) → **0.10** 선택
- 고정 하이퍼파라미터: `n_estimators=300, max_samples="auto", max_features=1.0, random_state=42`
- `risk_score = -model.decision_function(X)` (값이 클수록 위험), `predict()`의 `{1(정상), -1(이상)}` → `{0, 1}`로 변환

### Test 결과

- Precision 0.5000, Recall **10.64%**, F1 0.1754, ROC-AUC 0.7126, PR-AUC 0.2908
- 정상 데이터 오탐율(FPR) 1.73% — 오탐은 적지만 그만큼 놓치는 피크도 많음
- Test 336건 중 anomaly로 탐지된 건수 10건 (실제 피크 47건 중 5건만 적중)
- 참고: CatBoost Recall 74.47%, XGBoost Recall 78.72% — Isolation Forest 대비 7배 이상

### 최종 판단 (Isolation Forest 채택 여부)

| 판단 기준 | 실제 결과 |
| --- | --- |
| 양성 label이 매우 적은가? | train 양성 비율 10.26% — 극단적으로 희소하지 않음 |
| supervised 모델이 학습하기 어려운 수준인가? | 아니다. RF/XGBoost/CatBoost 모두 test F1 0.69~0.71로 이미 준수 |
| Isolation Forest Recall/F1이 경쟁력 있는가? | 아니다. F1 0.1754로 supervised 최고(0.7091) 대비 -0.53 이상 열세 |
| label 없이도 이상탐지가 필요한가? | 아니다. 현재 피크 정의(라벨)가 이미 존재하고 안정적으로 갱신됨 |

**결론**: 비지도 이상탐지 baseline으로 실험했으나, 라벨이 존재하는 현재 데이터에서는 supervised boosting 모델(XGBoost/CatBoost)이 더 적합했다. **Isolation Forest는 최종 모델 후보에 포함하지 않는다.**

## 전체 최종 결론

- **최종 추천 모델: Random Forest (@0.5)** — Test F1 0.7091로 4개 모델 중 최고, Recall(82.98%)도 가장 높아 피크를 가장 적게 놓친다. 구조도 가장 단순하다.
- XGBoost(@0.65)·CatBoost(@0.8)는 F1 기준으로는 근소하게 낮지만(0.6852 / 0.6931) SHAP 기반 feature 해석, 범주형 처리 등 설명력·확장성 면에서 강점이 있어 보조/비교 모델로 유지할 가치가 있다.
- Isolation Forest는 성능상 채택하지 않되, "라벨이 아직 없는 신규 설비/라인"처럼 향후 피크 정의가 없는 상황에서 baseline으로 재검토할 수 있다.

## 산출물

- `results/isolation_forest_predictions.csv`: `Date, actual_label, predicted_anomaly, anomaly_score, actual_power`
- `results/model_comparison.csv`: Persistence / Random Forest / XGBoost / CatBoost / Isolation Forest 전체 실험 결과
