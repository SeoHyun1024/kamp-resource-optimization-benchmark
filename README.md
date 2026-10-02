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
├─ ablation_encoding.py         # (선택) RF·XGBoost 달력 변수 인코딩 비교 실험
├─ ablation_fp_features.py      # (선택) 헛경보(FP) 원인 변수 실험
├─ 08_operating_calendar.py     # 에이전트 M1: 실제 전력 기준 가동 캘린더
├─ 09_peak_alert.py             # 에이전트 M2: RNN·RF 결합 2단계 피크 경보
├─ requirements.txt
└─ results/                     # 실행하면 자동 생성
   ├─ 1_preprocessing/          # 전처리 결과
   ├─ 2_test_predictions/       # 테스트데이터 예측결과 (제출용)
   ├─ 3_comparison/             # 모델 성능 비교표
   ├─ 4_model_details/          # CV 탐색표·OOF 예측·변수 중요도
   ├─ 5_models/                 # 학습된 모델 파일 + 메타데이터 json (모델 7종)
   └─ 6_figures/                # 그림
```

## 실행 방법

Python 3.11 기준입니다.

```bash
# 0. 패키지 설치 (처음 한 번)
pip install -r requirements.txt

# 1. 전처리
python preprocessing.py                               # → results/1_preprocessing/

# 2. 모델 학습·평가 (01~06은 서로 독립)
python 01_rnn.py --variant D_multi_gru --epochs 90   # 약 10~15분 (옵션 없이 실행하면 구조 4종 CV 포함 약 1시간)
python 02_random_forest.py                            # 약 2분
python 03_xgboost.py                                  # 약 30초
python 04_catboost.py                                 # 약 1분
python 05_isolation_forest.py                         # 약 10초
python 06_regression.py                               # 약 1~2분

# 3. 전체 비교 (반드시 마지막)
python 07_compare.py                                  # → results/3_comparison/final_comparison.csv

# 4. (선택) 보조 실험
python ablation_copied_days.py   # 복제일 처리 비교, 약 1분
python ablation_missing.py       # 결측 처리 비교, 약 4분
python ablation_encoding.py      # 달력 인코딩 비교, 약 1분
python ablation_fp_features.py   # 헛경보 원인 변수 실험, 약 3분

# 5. 에이전트 모듈 (01·02 실행 후)
python 08_operating_calendar.py  # M1 가동 캘린더, 수 초
python 09_peak_alert.py          # M2 피크 경보, 수 초 (01·02 결과와 08 결과 사용)
```

- 다시 실행하면 `results/`의 같은 이름 파일은 덮어써집니다. `07_compare.py`는 02~04의 결과로 앙상블을 만들므로 앞 스크립트를 다시 돌렸다면 함께 다시 실행합니다.

- 모든 스크립트는 `kamp` 폴더에서 실행합니다.
- 시드는 42로 고정했습니다. 다만 CPU나 라이브러리 버전에 따라 소수점 수준의 차이는 날 수 있습니다.

## 전처리

원자료를 진단한 결과와 처리 방법입니다. 실행 로그는 `results/1_preprocessing/preprocessing_report.txt`에 남습니다.

| 항목 | 진단 결과 | 처리 |
|---|---|---|
| 결측치 | 풍속 3건, 강수량 1건, 공장인원 17건 | 풍속·강수량은 시간 보간. 공장인원은 결측 행이 전부 생산량 0 구간임을 확인한 뒤 0으로 채움. 보간·직전값·0·모델 내장 처리 4가지를 비교했는데 성능 차이가 없어 보간을 유지함(`ablation_missing.py`) |
| `시간` 이상치 | 2021-07-13, 07-15 두 날의 48행이 0~23 범위를 벗어남(70~188) | 두 날 모두 24행이 온전하므로 날짜 안의 행 순서로 0~23시를 재구성 |
| 증강 복제일 | 257일 중 115일이 앞선 날의 15분 값 96개와 완전히 같음. 모두 1~7월 | 행은 유지하고 `is_copied_day` 플래그만 만듦. CV 검증 구간은 복제가 없는 7·8월로 잡음. 학습에서 빼면 CV 성능이 떨어져(`ablation_copied_days.py`) 학습에는 포함 |
| 공휴일 | 1/1, 설, 3/1, 5/5, 5/19, 8/16이 모두 가동일(일 최대전력 108~195) | 공휴일 변수를 쓰지 않음 |
| 주말 | 토요일 37일 중 35일 가동 | 주말 변수 대신 `day_of_week` 사용 |
| `month` | test(9월)가 학습 데이터에 없는 값 | 사용하지 않음 |
| 달력 인코딩 | 정수·one-hot·sin/cos·정수+sin/cos를 RF·XGBoost로 비교. one-hot이 가장 나쁨(CV PR-AUC −0.003~−0.014) | 모델별로 선택: RF 정수+sin/cos, XGBoost 정수, CatBoost 범주형, RNN sin/cos (`ablation_encoding.py`) |

**예측 대상**
- `power` = max(15분, 30분, 45분, 60분): 해당 시간의 최대수요전력입니다.
- 피크 라벨은 `power ≥ 179`입니다. 179는 2021-01-08~07-31 `power`의 90% 분위수예요.

## Feature (38개, 모두 T−1 시점까지의 정보)

| 그룹 | Feature |
|---|---|
| 전력 lag | `lag_1, 2, 3, 6, 12, 24, 48, 167, 168, 169, 336` |
| 직전 1시간 15분 단위 값 | `q15_lag1`, `q30_lag1`, `q45_lag1`, `q60_lag1`, `avg_lag1`, `intra_hour_slope`(60분 − 15분) |
| 추세·통계 | `diff_1`, `roll_mean_3`, `roll_mean_24`, `roll_max_24`, `roll_std_24`, `max_since_midnight` |
| 주간 패턴 | `same_hour_2w_mean`, `same_hour_2w_max`, `last_op_day_same_hour`(직전 가동일 같은 시각) |
| 전일 상태 | `prev_day_max`, `prev_day_off` |
| 외생변수(1시간 전) | 생산량, 기온, 풍속, 습도, 강수량, 전기요금, 공장인원, 인건비 |
| 달력 | `hour`, `day_of_week` (RF는 여기에 sin/cos 4개를 더해 42개) |

**누수 점검**: `preprocessing.leakage_checks()`에서 다음을 assert로 확인합니다.
- `lag_1`, `q60_lag1`, `production`이 T−1 값인지
- target과 같거나 상관계수가 0.99 이상인 feature가 없는지

## 모델별 입력 변환 (각 스크립트에서 수행)

모든 모델은 같은 분할·같은 피크 라벨(179)·같은 test(336시간, 피크 47건)를 씁니다. 데이터셋 파일은 공통 1개(`results/1_preprocessing/peak_dataset.csv`)이고, 모델에 맞춘 변환은 각 스크립트가 실행 중에 수행합니다(별도 파일로 저장하지 않음).

| 모델 | 스크립트 | 입력 | 수행하는 변환 |
|---|---|---|---|
| Random Forest | `02_random_forest.py` | 공통 feature 38개 + 4개 = **42개** | `hour`·`day_of_week` 정수에 sin/cos 4개(`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`)를 추가(`preprocessing.add_cyclic_calendar`). 스케일링 없음 |
| XGBoost | `03_xgboost.py` | 공통 feature **38개** 그대로 | 변환 없음(달력은 정수). `scale_pos_weight`는 학습 구간마다 음성/양성 비율로 다시 계산해 후보로 비교(최종 미사용) |
| CatBoost | `04_catboost.py` | 공통 feature **38개** | `hour`·`day_of_week`를 문자열로 바꿔 `cat_features`(범주형)로 지정. 나머지는 수치 그대로 |
| Isolation Forest | `05_isolation_forest.py` | 전력 관련 feature **21개** | 달력·외생변수를 빼고 전력 lag·15분 값·rolling·주간 패턴만 사용. 학습은 가동 시간대(전일 가동 & 08~18시) 행으로만 함. 점수는 `-decision_function`이며, `lag_1`이 학습 중앙값보다 낮은 시점의 이상은 피크 위험이 아니므로 최저 점수로 보정. 판정 경계는 학습 점수 상위 40% |
| XGBoost·LightGBM 회귀 | `06_regression.py` | 공통 feature **38개** | 라벨 대신 연속값 `target_power`를 예측 대상으로 사용. 예측값 ≥ cutoff(XGB 171, LGBM 173, CV로 선택)이면 피크로 판정 |
| RNN (GRU) | `01_rnn.py` | 시퀀스 **168시간 × 8채널** | `peak_dataset.csv`를 쓰지 않고 같은 정제 과정을 스스로 수행한 뒤, 예측 시점 T마다 T−168~T−1 구간을 잘라 시퀀스를 만듦. 채널은 power, 15분 값, 60분 값, 생산량, 시간 sin/cos, 요일 sin/cos. 입력·target 모두 MinMax 정규화. 예측 시점은 다른 모델과 같은 2021-01-15부터. 예측값 ≥ 167이면 피크로 판정 |

**변환에서 지킨 원칙**
- 학습 데이터로 무언가를 맞추는 변환은 **학습 구간으로만** 맞춥니다. MinMax 스케일러, Isolation Forest의 중앙값·판정 경계, 분류 threshold·회귀 cutoff가 여기에 해당하고, CV에서는 fold마다 다시 맞춥니다.
- sin/cos, 범주형 지정, feature 선택처럼 값을 학습하지 않는 변환은 데이터 전체에 같은 규칙으로 적용합니다.
- 모델별 선택의 근거:
  - 달력 인코딩: `ablation_encoding.py`
  - Isolation Forest 입력 구성: `05_isolation_forest.py`의 실험 A~C
  - RNN 입력 구조: `01_rnn.py`의 구조 A~D 비교

## 검증 방식

- **CV**: 확장창 2-fold로 hyperparameter, threshold, 모델 구조를 선택합니다.
  - fold 1: 7/1 이전으로 학습하고 7월로 검증
  - fold 2: 8/1 이전으로 학습하고 8월로 검증
- **최종 학습**: 9/1 이전 전체 데이터로 다시 학습합니다.
- **Test**: 2021-09-01~09-14(336시간, 피크 47건)이며, 선택에는 쓰지 않고 마지막에 1회만 평가합니다.

## 결과 (Test 2021-09-01~09-14)

| 모델 | CV F1 | CV PR-AUC | Test F1 (95% CI) | Test PR-AUC | Test Recall |
|---|---|---|---|---|---|
| 규칙: 직전 값 (lag_1 ≥ 179) | – | – | 0.447 | 0.529 | 0.447 |
| 규칙: 1주 전 같은 시각 (lag_168 ≥ 179) | – | – | 0.653 | 0.561 | 0.702 |
| Random Forest | 0.821 | 0.892 | 0.729 [0.58, 0.83] | 0.767 | 0.915 |
| XGBoost | 0.821 | 0.895 | 0.697 [0.57, 0.79] | 0.723 | 0.809 |
| CatBoost | 0.827 | 0.905 | 0.679 [0.54, 0.78] | 0.738 | 0.809 |
| 앙상블 (RF+XGB+CB 평균) | 0.829 | 0.911 | 0.692 [0.53, 0.80] | 0.742 | 0.787 |
| RNN (GRU, 예측값 ≥ 167) | – | – | 0.732 [0.59, 0.83] | 0.744 | 0.957 |
| XGBoost 회귀 (예측값 ≥ 171) | 0.791 | – | 0.705 [0.54, 0.82] | 0.728 | 0.915 |
| LightGBM 회귀 (예측값 ≥ 173) | 0.806 | – | 0.690 [0.52, 0.80] | 0.685 | 0.851 |
| Isolation Forest (비지도) | 0.618 | 0.504 | 0.485 [0.37, 0.58] | 0.344 | 0.702 |

회귀 성능(Test MAE):

| 모델 | MAE | RMSE |
|---|---|---|
| 직전 값 (lag_1) | 14.52 | 23.78 |
| 1주 전 같은 시각 (lag_168) | 8.95 | 12.33 |
| XGBoost 회귀 | 6.12 | 8.61 |
| LightGBM 회귀 | 6.27 | 9.79 |
| RNN (GRU) | 6.72 | 9.13 |

**RNN 구조 선택 결과**: CV MAE 기준으로 GRU가 선택됐습니다.

| 구조 | 입력 형태 | CV MAE |
|---|---|---|
| A: SimpleRNN | (168, 1) | 11.48 |
| B: SimpleRNN | (7, 24) | 14.14 |
| C: SimpleRNN | (168, 8) | 9.21 |
| D: GRU | (168, 8) | **8.24** |

### 해석

1. **지도학습 모델은 모두 규칙 기반 기준선보다 좋습니다.** RF, XGBoost, CatBoost, RNN, 회귀 모델의 F1은 0.68~0.73입니다.
   - 규칙 기반 기준선은 직전 값 0.447, 1주 전 같은 시각 0.653입니다.
2. **모델 간 차이는 통계적으로 확정할 수 없습니다.**
   - test가 14일(피크 47건)뿐이라 F1 95% 신뢰구간 폭이 약 ±0.15입니다.
   - 신뢰구간은 일 단위 블록 부트스트랩(2000회)으로 구했습니다.
   - 그래서 모델을 고를 때는 CV 지표와 함께 판단합니다.
3. **test F1은 RNN(0.732)과 Random Forest(0.729)가 가장 높습니다.**
   - RNN은 Recall이 가장 높아(0.957, 놓친 피크 2건) 피크를 놓치지 않는 것이 중요한 경보 용도에 적합합니다. 대신 Precision이 0.592로 오탐이 많습니다.
   - RF는 Recall 0.915, Precision 0.606으로 RNN과 비슷한 성향이며, 학습 시간이 짧고 변수 중요도를 바로 볼 수 있습니다.
4. **Isolation Forest는 라벨 없이 동작하지만 성능이 낮습니다.**
   - F1 0.485로, 1주 전 같은 시각 규칙보다도 낮습니다.
   - 라벨이 없는 신규 설비용 참고 모델로만 둡니다.
5. **중요 변수**: 1·2주 전 같은 시각 전력(`same_hour_2w_max`), 직전 시간 마지막 15분 값(`q60_lag1`), 시간대(`hour`)가 모든 모델에서 상위권입니다.
   - 즉 공장의 **주간 가동 패턴**과 **직전 추세**가 피크를 가장 잘 설명합니다.

## 에이전트 확장 (진행 중)

피크 예측 결과를 운영 의사결정으로 잇기 위해 기능을 모듈로 나눠 만들고 있습니다. 판단은 규칙과 검증된 모델로 하고, 최종 실행은 담당자가 승인하는 구조입니다.

| 모듈 | 역할 | 스크립트 | 상태 |
|---|---|---|---|
| M1 가동 캘린더 | 실제 전력으로 가동일·날 유형 판정 | `08_operating_calendar.py` | 완료 |
| M2 피크 경보 | RNN(주의) + RF(확정) 2단계 경보, 근거 문구 생성 | `09_peak_alert.py` | 완료 |
| M3 전력 예측 | 가동 계획을 넣은 하루 전 전력 예측(회귀) | – | 예정 |
| M4 일정 조정 | 피크 시간 부하 이동 시 최대수요·기본요금 절감 계산 | – | 예정 |
| M5 추천 리포트 | M1~M4 결과를 운영 리포트로 출력 | – | 예정 |

### M1 가동 캘린더

- 판정 기준은 9/1 이전 **원본 일자**에서 계산합니다. 일 최대전력이 비가동일 최대 41, 가동일 최소 104로 나뉘어 가동일 기준은 72.5입니다. 시간 단위로는 41을 넘으면 가동 중으로 봅니다.
- 날 유형: 종일 가동, 가동 시작일(0시 꺼짐 → 이후 가동), 가동 종료일(23시 꺼짐), 중간 정지, 비가동
- 257일 중 115일은 다른 날짜의 전력을 복사한 증강일입니다. 증강일은 달력·생산 기록과 비교하면 왜곡되므로(예: 2/11 공휴일 = 1/11 월요일 가동 패턴 복사) **원본 일자를 주 결과로** 보고합니다.

| 지표 | 원본 (주 결과) | 증강 | 전체 |
|---|---|---|---|
| 일수 | 142 | 115 | 257 |
| 생산 기록 일치율 | 93.7% (불일치 9일) | 94.8% (6일) | 94.2% (15일) |
| 공휴일 가동 | 4/4 | 3/3 | 7/7 |
| 토요일 가동 | 20/22 | 15/15 | 35/37 |
| 일요일 가동 | 6/24 | 2/13 | 8/37 |
| 평일 비가동 | 6/92 | 12/84 | 18/176 |
| 달력과 다른 날 | 36 (25.4%) | 32 (27.8%) | 68 (26.5%) |

- 원본 공휴일 4일(1/1, 5/5, 5/19, 8/16)이 모두 가동했고, 토요일은 대부분 오전에 끝나는 가동 종료일, 월요일은 7시에 켜지는 가동 시작일입니다.
- 원본 평일 휴무는 1/5와 하계 휴무(8/2~8/6) 6일뿐입니다. 증강일의 평일 비가동 12일은 모두 1/5 패턴의 복사본입니다.
- 생산 기록 불일치일에는 시간 값이 깨져 있던 7/13, 7/15가 포함되어, 두 날의 데이터 품질 문제가 다시 확인됩니다.

### M2 피크 경보

| 단계 | 조건 | 목적 |
|---|---|---|
| 주의 | RNN 예측 전력 ≥ 167 | 피크를 놓치지 않기 |
| 확정 | 주의 + RF 위험확률 ≥ 0.55 | 실제 조치 대상 |

기준값은 `5_models/*_meta.json`에서 읽습니다. 전환 시각(13·15·16·17시) 경보에는 헛경보 확인 안내가 붙습니다.

| 단계 (test 14일, 피크 47건) | 경보 | 맞힌 피크 | 헛경보 | 놓친 피크 | Precision | Recall | F1 | 하루당 헛경보 |
|---|---|---|---|---|---|---|---|---|
| 주의 이상 | 76 | 45 | 31 | 2 | 0.592 | 0.957 | 0.732 | 2.2 |
| 확정 | 69 | 43 | 26 | 4 | 0.623 | 0.915 | 0.741 | 1.9 |
| 참고: RF 단독 | 71 | 43 | 28 | 4 | 0.606 | 0.915 | 0.729 | 2.0 |

- RF만 위험인 시간은 2건이며 모두 피크가 아니어서 정상으로 둡니다.
- 확정 규칙(RF AND RNN)은 test 결과를 보고 제시한 후보입니다. RNN은 OOF 예측이 없어 CV로 검증하지 못했으므로, 운영 데이터로 재검증해야 합니다.

### 헛경보(FP) 원인 분석 (`ablation_fp_features.py`)

- CV(7·8월)에서 RF 헛경보는 가동 전환 시각(13시 점심 직후, 15~17시 퇴근 전후)에 약 57%가 몰립니다.
  - 15~17시: 직전 시간이 이미 피크여서 모델이 피크가 이어진다고 보지만, 실제로는 내려가며 170~178에 머뭅니다.
  - 13시: 직전 값(점심 시간)이 낮아 정보가 없고 2주 전 같은 시각 최대에 의존합니다.
- 헛경보의 약 64%는 실제 전력 170~178의 경계 사례입니다.
- 시간 내 하락, 피크 지속, 시각별 전환 패턴, 생산 변화 변수(G1~G4)를 추가하면 RF CV 헛경보가 약 14% 줄지만 놓친 피크가 늘고, test에서는 개선이 없었습니다. 교대·점심 시각 같은 작업 스케줄 정보가 데이터에 없어서이며, 향후 과제로 둡니다.

## 결과 파일 (`results/`)

| 폴더 | 파일 | 내용 |
|---|---|---|
| `1_preprocessing/` | `clean_hourly.csv` | 정제된 시간 단위 데이터 |
| | `operating_calendar.csv` | M1: 날짜별 가동 여부·날 유형·가동 시간·생산 기록·달력 구분·증강일 여부 |
| | `peak_dataset.csv` | 모델 공통 feature 38개 + target + 라벨 |
| | `copied_days.json` | 복제일 → 원본 날짜 |
| | `preprocessing_report.txt` | 전처리 진단 로그 |
| `2_test_predictions/` | `rnn_forecast.csv` | RNN: `Date, actual, forecast, actual_label, predicted_label` |
| | `random_forest_predictions.csv` | `Date, actual_label, predicted_label, risk_probability, actual_power` |
| | `xgboost_predictions.csv`, `catboost_predictions.csv`, `ensemble_rf_xgb_cb_predictions.csv` | 같은 형식 |
| | `xgboost_reg_predictions.csv`, `lightgbm_reg_predictions.csv` | 회귀: `predicted_power` 포함 |
| | `isolation_forest_predictions.csv` | `anomaly_score` 포함 |
| | `persistence_predictions.csv`, `persistence_lag168_predictions.csv` | 규칙 기반 기준선 |
| | `peak_alerts.csv` | M2: 시간별 경보 단계, RNN 예측값, RF 확률, 근거 문구, 전환 시각 안내, 날 유형 |
| `3_comparison/` | `final_comparison.csv` | **전체 모델 test 성능 + F1 95% 신뢰구간 (보고서용)** |
| | `model_comparison.csv` | 모델별 test 성능 |
| | `regression_comparison.csv` | 회귀 MAE/RMSE |
| | `rnn_metrics.txt` | RNN test 성능 요약 |
| | `missing_ablation_summary.csv` | 결측 처리 방식별 성능 비교 요약 (`ablation_missing.py`) |
| | `encoding_ablation_summary.csv` | 달력 인코딩별 성능 비교 요약 (`ablation_encoding.py`) |
| | `fp_feature_ablation.csv` | 헛경보 원인 변수 실험 결과 (`ablation_fp_features.py`) |
| | `operating_calendar_summary.csv`, `operating_calendar_types.csv` | M1 평가지표(원본·증강·전체), 날 유형 × 요일 표(원본) |
| | `peak_alert_summary.csv` | M2 경보 단계별 성능 |
| `4_model_details/` | `*_cv.csv`, `*_oof.csv`, `*_importance.csv` | grid 탐색 결과, CV out-of-fold 예측, 변수 중요도 |
| | `rnn_cv.csv` | RNN 구조별 CV 결과 (CV 포함 실행 시) |
| | `missing_ablation_real.csv`, `missing_ablation_simulated.csv`, `encoding_ablation.csv` | 결측 처리·인코딩 비교 상세 |
| | `operating_calendar_mismatch.csv` | M1 생산 기록 불일치일 목록 (증강일 여부·원본 날짜 포함) |
| | `peak_alert_daily.csv` | M2 날짜별 피크·주의·확정·놓친 피크 수 |
| `5_models/` | `random_forest.joblib`, `xgboost_classifier.json`, `catboost_classifier.cbm`, `isolation_forest.joblib`, `xgboost_reg.joblib`, `lightgbm_reg.joblib`, `rnn.keras` (+ `rnn_scalers.joblib`) | 9/1 이전 전체로 학습한 최종 모델 |
| | `*_meta.json` | 모델별 메타데이터: 판정 기준(threshold·cutoff), feature 목록과 순서, 파라미터, 학습·test 기간, 피크 기준 179, CV·test 성능 |
| `6_figures/` | `rnn_forecast_plot.png` | RNN 테스트 예측 그래프 |
| | `operating_calendar.png` | M1 가동 캘린더 |
| | `peak_alert_timeline.png` | M2 test 기간 실제 전력과 경보 시점 |

## 저장된 모델 불러오기

```python
import json, joblib
meta = json.load(open("results/5_models/random_forest_meta.json", encoding="utf-8"))
rf = joblib.load("results/5_models/" + meta["model_file"])
proba = rf.predict_proba(X[meta["features"]])[:, 1]          # X = feature 표 (meta["features"] 순서)
peak = proba >= meta["classification_threshold"]              # 피크 위험 판정
```

- XGBoost는 `XGBClassifier().load_model(...)`, CatBoost는 `CatBoostClassifier().load_model(...)`로 불러와요.
- RNN은 `keras.models.load_model("results/5_models/rnn.keras")`로 불러오고, 입력·출력 변환에는 `rnn_scalers.joblib`를 써요.