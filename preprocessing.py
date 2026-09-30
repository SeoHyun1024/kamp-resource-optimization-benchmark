"""
01 전처리 (초기 01_preprocessing.ipynb 재작성) + 모델 공통 데이터셋 생성

실행:  python preprocessing.py
산출:  results/1_preprocessing/clean_hourly.csv        (정제된 시간 단위 원자료)
       results/1_preprocessing/peak_dataset.csv        (모델 공통 feature + target + split 정보)
       results/1_preprocessing/copied_days.json        (복제일 -> 원본 날짜)
       results/1_preprocessing/preprocessing_report.txt

이 파일은 02~06 모델 스크립트가 import 해서 쓰는 "공통 모듈"이기도 하다.
(load_dataset(), cv_folds(), final_split(), evaluate() 등)

------------------------------------------------------------------------------
초기 노트북 대비 수정 사항 (근거는 전부 이 파일 실행 로그로 재확인 가능)
------------------------------------------------------------------------------
[1] 결측치: df.fillna(0) 일괄 처리 -> 컬럼별 처리
    - 풍속(3건)·강수량(1건)을 0으로 채우면 "무풍/무강수"라는 가짜 값이 된다.
      -> 시계열 선형 보간 (baseline_rf.py 와 동일한 방식)
    - 공장인원(17건)은 결측 행이 전부 생산량=0 구간(8/28~8/29)임을 assert로 확인한 뒤 0.

[2] "시간" 이상치(2021-07-13, 07-15, 48행, 값 70~188)
    - 01 노트북은 date_range로 index만 새로 만들고 "시간" 컬럼은 깨진 채로
      df_frame.csv에 저장했다 -> 02_random_forest 노트북이 그 "시간"을 feature로
      써서 고유값이 60개가 됐다(정상은 24개).
    - 날짜 안 행 순서로 0~23시를 재구성(data_quality_metrics.py 와 동일)한 뒤 저장.

[3] 증강(복제) 데이터 탐지  ※ 가장 큰 문제
    - 257일 중 115일(45%)이 "그 이전 어떤 날의 15분 값 96개와 완전히 동일"하다.
      전부 1~7월에 있고, 8월·9월(검증/테스트)에는 없다.
    - 행 단위 유일성(100%)으로는 안 보이고, 일 단위로 봐야 드러난다.
    - 복제일은 요일·생산량 등 다른 컬럼과 전력 패턴이 어긋난다
      (예: 화·수요일인데 일요일 같은 비가동 패턴). 모델이 "요일 -> 전력" 관계를
      잘못 배우는 원인이 된다.
    - 처리: 행은 지우지 않는다(lag가 끊기지 않도록). is_copied_day 플래그를 만들어
      (a) CV 검증 구간에서는 제외하고 (b) 학습에서 제외/가중치 축소를 실험했다.
      ablation_copied_days.py 결과(7·8월 CV, seed 3개 평균) 학습에서 빼면 오히려
      PR-AUC가 0.861 -> 0.843으로 떨어져(학습 데이터 절반 손실), 기본값은 "유지"다.
      -> 복제일은 "지워야 할 행"이 아니라 "1~7월로 검증하면 점수가 부풀려지는 이유"로
         다루고, 그래서 CV 검증 구간을 복제가 없는 7·8월로 잡았다.

[4] 달력 변수 재검토
    - 01 노트북의 공휴일 7일(1/1, 2/11, 2/12, 3/1, 5/5, 5/19, 8/16)은 이 데이터에서
      전부 가동일(일 최대전력 108~195)이다 -> Vacation 플래그가 오히려 잘못된 정보.
    - 토요일은 대부분 가동(일 최대전력 중앙값 115)하므로 Weekend(토+일)=1은
      "쉬는 날"이라는 의미가 아니다.
    - month는 test(9월)가 train에 없는 값이라 외삽만 일으킨다(CatBoost 노트북 12-2절).
    -> 세 변수 제거. 대신 실제 전력으로 판정한 "전일 가동 여부/직전 가동일 같은 시각"
       을 쓴다(과거 정보만 사용 -> 누수 없음).

[5] 예측 대상 정의 통일
    - 03~06은 target = max(15분,30분,45분,60분), 01 RNN은 target = 15분.
      서로 다른 값이라 MAE를 직접 비교하면 안 된다. 여기서는 03~06 정의를 따른다.

[6] 검증 방식
    - 기존: 8월 한 달(하계 휴무 8/2~8/8 포함)만으로 튜닝+threshold 선택 -> 9월과 분포가
      달라 threshold가 흔들렸다(XGB 0.65, CatBoost 0.8).
    - 변경: 시간순 확장창(rolling-origin) 2-fold CV (7월, 8월 검증) -> 두 fold의
      out-of-fold 예측을 합쳐 hyperparameter·threshold를 고른다.
      최종 모델은 9/1 이전 전체로 재학습하고 test(9/1~9/14)는 1회만 평가한다.
    - 피크 임계값은 기존과 같은 정의(2021-01-08~07-31 target의 90% quantile = 179)로
      고정해, test 라벨(피크 47건)이 기존 결과와 완전히 같게 유지한다(공정 비교).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

DATA_PATH = "./data/okm_augumented_2021.csv"
RESULTS_DIR = "results"
OUT_DIRS = {                     # results/ 하위 폴더 구성
    "prep": "1_preprocessing",     # 전처리 결과
    "pred": "2_test_predictions",  # 테스트데이터 예측결과 (제출용)
    "cmp": "3_comparison",         # 모델 성능 비교표
    "detail": "4_model_details",   # CV 탐색표·OOF 예측·변수 중요도
    "model": "5_models",           # 학습된 모델 파일
    "fig": "6_figures",            # 그림
}


def out(kind: str, name: str) -> str:
    """results/<하위폴더>/<파일명> 경로를 만들고(폴더 자동 생성) 반환."""
    d = os.path.join(RESULTS_DIR, OUT_DIRS[kind])
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)

SEED = 42
QUARTERS = ["15분", "30분", "45분", "60분"]

# 분할 경계
LEGACY_TRAIN_START = pd.Timestamp("2021-01-08")  # 기존 노트북 train 시작(lag_168 이후)
LEGACY_VAL_START = pd.Timestamp("2021-08-01")    # 기존 노트북 validation 시작
TEST_START = pd.Timestamp("2021-09-01")          # test: 2021-09-01 ~ 09-14 (336시간)
CV_FOLDS = [                                     # (검증 시작, 검증 끝) - 검증 이전 전체가 학습
    (pd.Timestamp("2021-07-01"), pd.Timestamp("2021-08-01")),
    (pd.Timestamp("2021-08-01"), pd.Timestamp("2021-09-01")),
]

PEAK_QUANTILE = 0.90
OFF_DAY_MAX_POWER = 60        # 일 최대전력 < 60 이면 비가동일 (가동일 최소 108, 비가동일 최대 41)
DROP_COPIED_FROM_TRAIN = False  # CV 실험 결과 유지가 더 좋음 (ablation_copied_days.py)
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)

LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 167, 168, 169, 336]
EXOG_RENAME = {
    "생산량": "production", "기온": "temperature", "풍속": "wind_speed",
    "습도": "humidity", "강수량": "precipitation", "전기요금(계절)": "elec_price",
    "공장인원": "workers", "인건비": "labor_cost",
}
CATEGORICAL = ["hour", "day_of_week"]


# =============================================================================
# 단계 1~3: 불러오기 / 확인 / 정제
# =============================================================================
def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


MISSING_STRATEGIES = {
    "interp": "시간 보간(앞뒤 값) + 공장인원 0  <- 기본값",
    "ffill": "직전 관측값 유지 + 공장인원 0 (실시간 운영에서 그대로 재현 가능한 방식)",
    "zero": "전부 0 (기존 01 노트북 fillna(0))",
    "native": "결측 그대로 두고 모델 내장 결측 처리 사용 (XGBoost·CatBoost·RF 지원)",
}
EXOG_WITH_NA = ["풍속", "강수량", "공장인원"]


def clean(raw: pd.DataFrame, log=print, missing: str = "interp") -> pd.DataFrame:
    df = raw.copy()
    log(f"[정보] 원자료 shape: {df.shape}")

    # --- [2] 시간 이상치 보정 -------------------------------------------------
    rows_per_day = df.groupby("날짜").size()
    assert (rows_per_day == 24).all(), "하루 24행이 아닌 날짜가 있음"
    broken = df.loc[~df["시간"].between(0, 23), "날짜"].unique().tolist()
    df["시간"] = df.groupby("날짜").cumcount()
    log(f"[정보] '시간' 이상치 날짜 {broken} -> 날짜 내 행 순서로 0~23시 재구성")

    ts = pd.to_datetime(df["날짜"].astype(str), format="%Y%m%d") + pd.to_timedelta(df["시간"], unit="h")
    df = df.set_index(pd.DatetimeIndex(ts, name="Date")).sort_index()
    assert df.index.is_unique
    assert (df.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all(), "1시간 간격 아님"
    assert (df["day"] == df.index.dayofweek + 1).all(), "day 컬럼과 실제 요일 불일치"

    # --- [1] 결측치 --------------------------------------------------------------
    na_before = df.isna().sum()
    log(f"[정보] 결측치(보정 전): {na_before[na_before > 0].to_dict()}")
    assert missing in MISSING_STRATEGIES, missing
    na_cols = [c for c in df.columns if df[c].isna().any()]
    idle_workers = df["공장인원"].isna() & (df["생산량"] == 0)
    log(f"[정보] 공장인원 결측 {int(df['공장인원'].isna().sum())}건 중 생산량 0 구간 {int(idle_workers.sum())}건")
    if missing != "native":
        df.loc[idle_workers, "공장인원"] = 0.0  # 생산이 없으면 투입 인원도 0 (원자료 17건 모두 해당)
        if missing == "interp":
            df[na_cols] = df[na_cols].interpolate(method="time", limit_direction="both")
        elif missing == "ffill":
            df[na_cols] = df[na_cols].ffill().bfill()
        elif missing == "zero":
            df[na_cols] = df[na_cols].fillna(0)
        assert df.isna().sum().sum() == 0
    # native: 그대로 둔다 (모델이 결측을 직접 처리)
    log(f"[정보] 결측 처리 = {missing}: {MISSING_STRATEGIES[missing]}")

    # --- [5] 예측 대상 전력 -------------------------------------------------------
    df["power"] = df[QUARTERS].max(axis=1).astype(float)
    gap = (df["평균"] - df[QUARTERS].mean(axis=1)).abs().max()
    log(f"[정보] power = max(15/30/45/60분). '평균' = 4개 평균의 반올림값(최대 오차 {gap})")

    # --- [3] 복제일 탐지 ---------------------------------------------------------
    day_key = df.groupby("날짜")[QUARTERS].apply(lambda g: tuple(g.to_numpy().ravel()))
    first_seen, copy_of = {}, {}
    for d, key in day_key.items():
        if key in first_seen:
            copy_of[d] = first_seen[key]
        else:
            first_seen[key] = d
    df["is_copied_day"] = df["날짜"].isin(copy_of).astype(int)
    by_month = pd.Series(list(copy_of)).floordiv(100).mod(100).value_counts().sort_index()
    log(f"[정보] 이전 날짜와 15분 값 96개가 완전히 같은 '복제일': {len(copy_of)}일 / {len(day_key)}일")
    log(f"       월별: {by_month.to_dict()}  (8·9월 0일)")
    df.attrs["copy_of"] = copy_of

    # --- [4] 가동일/달력 진단 ----------------------------------------------------
    day_max = df.groupby("날짜")["power"].max()
    off_days = day_max[day_max < OFF_DAY_MAX_POWER]
    log(f"[정보] 비가동일(일 최대 < {OFF_DAY_MAX_POWER}): {len(off_days)}일, "
        f"가동일 최소 {day_max[day_max >= OFF_DAY_MAX_POWER].min():.0f} / 비가동일 최대 {off_days.max():.0f}")
    holidays = [20210101, 20210211, 20210212, 20210301, 20210505, 20210519, 20210816]
    log(f"[정보] 01 노트북 공휴일 일 최대전력: {day_max.loc[holidays].astype(int).to_dict()} -> 전부 가동일")
    dow = df.groupby("날짜")["day"].first()
    sat_max = day_max[dow == 6]
    log(f"[정보] 토요일 일 최대전력 중앙값 {sat_max.median():.0f} (비가동 {int((sat_max < OFF_DAY_MAX_POWER).sum())}/{len(sat_max)}일)")
    return df


# =============================================================================
# 단계 3-2: feature 구성 (행 index = 예측 대상 시점 T, feature는 T-1 이하 정보만)
# =============================================================================
def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    p = df["power"]
    past = p.shift(1)  # t = T-1 까지의 전력
    idx = df.index
    date = idx.normalize()

    # 일 단위 요약(전일 값만 사용): 어제 날짜의 최대전력 / 가동 여부
    daily_max = p.groupby(date).max()
    prev_day_max = pd.Series(daily_max.shift(1).reindex(date).to_numpy(), index=idx)

    # 직전 "가동일"의 같은 시각 전력 (1~7일 전 중 가장 가까운 가동일)
    op_day = daily_max >= OFF_DAY_MAX_POWER
    last_op_same_hour = pd.Series(np.nan, index=idx)
    for k in range(7, 0, -1):  # 먼 날부터 채우고 가까운 날로 덮어쓴다
        day_k = date - pd.Timedelta(days=k)
        ok = pd.Series(op_day.reindex(day_k).to_numpy(), index=idx).fillna(False).astype(bool)
        last_op_same_hour = last_op_same_hour.mask(ok, p.shift(24 * k))
    last_op_same_hour = last_op_same_hour.fillna(p.shift(168))  # 7일 내 가동일이 없으면 1주 전 값

    # 오늘 0시 ~ t 까지의 최대전력 (T가 0시면 오늘 정보 없음 -> 0)
    max_since_midnight = p.groupby(date).cummax().shift(1)  # 같은 날 0시 ~ T-1시 최대
    max_since_midnight[idx.hour == 0] = 0.0

    lag = {f"lag_{k}": p.shift(k) for k in LAG_HOURS}
    feats = pd.concat(
        [
            pd.DataFrame(lag),
            df[QUARTERS + ["평균"]].shift(1).set_axis(["q15_lag1", "q30_lag1", "q45_lag1", "q60_lag1", "avg_lag1"], axis=1),
            pd.DataFrame({
                "intra_hour_slope": df["60분"].shift(1) - df["15분"].shift(1),  # t시간 안의 추세
                "diff_1": lag["lag_1"] - lag["lag_2"],
                "roll_mean_3": past.rolling(3).mean(),
                "roll_mean_24": past.rolling(24).mean(),
                "roll_max_24": past.rolling(24).max(),
                "roll_std_24": past.rolling(24).std(),
                "same_hour_2w_mean": (lag["lag_168"] + lag["lag_336"]) / 2,
                "same_hour_2w_max": np.maximum(lag["lag_168"], lag["lag_336"]),
                "last_op_day_same_hour": last_op_same_hour,
                "prev_day_max": prev_day_max,
                "prev_day_off": (prev_day_max < OFF_DAY_MAX_POWER).astype(int),
                "max_since_midnight": max_since_midnight,
            }, index=idx),
            df[list(EXOG_RENAME)].shift(1).rename(columns=EXOG_RENAME),
            pd.DataFrame({"hour": idx.hour, "day_of_week": idx.dayofweek}, index=idx),
        ],
        axis=1,
    )
    features = list(feats.columns)
    # lag가 없는 앞부분만 제거한다. 외생변수 결측(missing="native")은 남겨 모델이 처리하게 한다.
    exog_cols = list(EXOG_RENAME.values())
    ds = feats.assign(target_power=p, is_copied_day=df["is_copied_day"]).dropna(
        subset=[c for c in features if c not in exog_cols])
    return ds, features


def peak_threshold_legacy(df: pd.DataFrame) -> float:
    """기존 노트북과 같은 정의: 2021-01-08 ~ 07-31 target의 90% quantile."""
    window = df.loc[(df.index >= LEGACY_TRAIN_START) & (df.index < LEGACY_VAL_START), "power"]
    return float(window.quantile(PEAK_QUANTILE))


def leakage_checks(ds: pd.DataFrame, df: pd.DataFrame, features: list[str]) -> None:
    prev = ds.index - pd.Timedelta(hours=1)
    assert np.array_equal(ds["lag_1"].to_numpy(), df.loc[prev, "power"].to_numpy()), "lag_1 != power(T-1)"
    assert np.array_equal(ds["q60_lag1"].to_numpy(), df.loc[prev, "60분"].to_numpy())
    assert np.array_equal(ds["production"].to_numpy(), df.loc[prev, "생산량"].to_numpy(), equal_nan=True)
    same = [c for c in features if np.array_equal(ds[c].to_numpy(), ds["target_power"].to_numpy())]
    assert not same, f"target과 동일한 feature: {same}"
    corr = ds[features].corrwith(ds["target_power"]).abs()
    assert (corr < 0.99).all(), corr.sort_values().tail()


# =============================================================================
# 모델 스크립트용 데이터 로더 (CV·평가 함수는 common.py)
# =============================================================================
def load_dataset(log=print, missing: str = "interp") -> tuple[pd.DataFrame, list[str], float]:
    df = clean(load_raw(), log=log, missing=missing)
    ds, features = build_features(df)
    leakage_checks(ds, df, features)
    thr = peak_threshold_legacy(df)
    ds["label"] = (ds["target_power"] >= thr).astype(int)
    return ds, features, thr


# =============================================================================
if __name__ == "__main__":
    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        lines.append(msg)

    raw = load_raw()
    df = clean(raw, log=log)
    ds, features, thr = load_dataset(log=lambda *_: None)

    log(f"\n[정보] feature {len(features)}개: {features}")
    log(f"[정보] lag_336 등으로 앞 {len(df) - len(ds)}행 제외 -> {len(ds)}행 ({ds.index.min()} ~ {ds.index.max()})")
    log(f"[정보] 피크 임계값 = {thr:.1f} (기존 정의와 동일)")
    from common import cv_folds, final_split
    tr, te = final_split(ds)
    log(f"[정보] 최종 train {len(tr)}행 (그중 복제일 {int(tr['is_copied_day'].sum())}행, 학습 제외={DROP_COPIED_FROM_TRAIN}) / test {len(te)}행, test 피크 {int(te['label'].sum())}건")
    for name, a, b in cv_folds(ds):
        log(f"[정보] CV fold {name}: train {len(a)}행(피크 {a['label'].mean():.1%}) / val {len(b)}행(피크 {b['label'].mean():.1%})")

    df.drop(columns=["is_copied_day"]).to_csv(out("prep", "clean_hourly.csv"))
    split = pd.Series("train", index=ds.index).mask(ds.index >= TEST_START, "test")
    ds.assign(split=split).to_csv(out("prep", "peak_dataset.csv"))
    with open(out("prep", "copied_days.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): str(v) for k, v in df.attrs["copy_of"].items()}, f, ensure_ascii=False, indent=1)
    with open(out("prep", "preprocessing_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n[정보] results/1_preprocessing/ 에 clean_hourly.csv, peak_dataset.csv, copied_days.json, preprocessing_report.txt 저장 완료")
