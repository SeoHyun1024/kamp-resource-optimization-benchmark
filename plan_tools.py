"""
주간 계획 공통 모듈 (13_week_ahead.py, 14_plan_advisor.py가 import)

1주 전 예측: 계획 주(최대 7일 = 168시간)의 시작 직전까지 쌓인 실제 전력만 쓴다.
    대상 시각 T의 전력 feature는 모두 T-168시간 이전 값이다. 그래서 계획 주가 어느 요일에 시작하든
    주 안의 모든 시각에서 미래 값이 섞이지 않는다.

계획 CSV 형식 (시간마다 한 행)
    date,hour,on,production
    2021-09-15,0,1,120
    ...
    - on         : 그 시간 가동 여부 (1/0)
    - production : 그 시간 계획 생산량 (데이터의 '생산량'과 같은 단위). 비우면 같은 요일·시각의 최근 평균으로 채운다.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import preprocessing as pp

HORIZON = 168                                  # 1주 전: T-168시간 이전 값만 사용
WEEK_LAGS = [168, 336, 504]
DAY_TYPES = ["비가동", "종일 가동", "가동 시작일", "가동 종료일", "중간 정지", "부분 가동"]
COMPANY_HOLIDAYS = pd.DatetimeIndex(["2021-01-05"]).append(pd.date_range("2021-08-02", "2021-08-08"))
PATTERN_MIN_SHARE = 0.5
TEMPLATE_WEEKS = 4                             # 템플릿 생산량: 최근 4주(원본 일자) 같은 요일·시각 평균
DOW_KO = ["월", "화", "수", "목", "금", "토", "일"]


# =============================================================================
# 데이터
# =============================================================================
def load_history() -> tuple[pd.DataFrame, dict, float, float]:
    """정제된 시간 단위 자료, 복제일 매핑, 시간 단위 가동 기준(41), 피크 기준(179)."""
    df = pp.clean(pp.load_raw(), log=lambda *_: None)
    copied = {pd.Timestamp(k): pd.Timestamp(v) for k, v in df.attrs.get("copy_of", {}).items()}
    summary = pd.read_csv(pp.out("cmp", "operating_calendar_summary.csv"), index_col=0, encoding="utf-8-sig")
    hour_thr = float(summary.loc["hour_threshold", "전체"])
    peak_thr = float(pp.peak_threshold_legacy(df))
    return df, copied, hour_thr, peak_thr


def executed_plan(df: pd.DataFrame, copied: dict, hour_thr: float) -> pd.DataFrame:
    """과거 기간의 '실제로 실행된 계획': 가동 = 전력 > 41, 생산량 = 기록.
    복제일은 전력이 원본 날짜의 것이라, 생산량도 원본 날짜 같은 시각의 값으로 맞춘다."""
    on = (df["power"] > hour_thr).astype(int)
    prod = df["생산량"].astype(float).copy()
    for day, src in copied.items():
        dst_idx = pd.date_range(day, periods=24, freq="h")
        src_idx = pd.date_range(src, periods=24, freq="h")
        if src_idx.isin(prod.index).all() and dst_idx.isin(prod.index).all():
            prod.loc[dst_idx] = prod.loc[src_idx].to_numpy()
    return pd.DataFrame({"on": on, "production": prod}, index=df.index)


# =============================================================================
# feature
# =============================================================================
def week_features(power: pd.Series, exog: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """index(예측 대상 시각)마다 T-168 이전 값으로 만든 feature. power는 과거 실제 전력(미래 칸은 NaN)."""
    full_idx = power.index.union(index)
    p = power.reindex(full_idx)
    date = full_idx.normalize()
    lag = {f"lag_{k}": p.shift(k, freq="h").reindex(full_idx) for k in WEEK_LAGS}

    # 7~13일 전 중 가장 가까운 '가동일'의 같은 시각 전력
    day_max = p.groupby(date).max()
    op_day = day_max >= pp.OFF_DAY_MAX_POWER
    last_op = pd.Series(np.nan, index=full_idx)
    for k in range(13, 6, -1):
        ok = pd.Series(op_day.reindex(date - pd.Timedelta(days=k)).to_numpy(), index=full_idx).fillna(False).astype(bool)
        last_op = last_op.mask(ok, p.shift(24 * k, freq="h").reindex(full_idx))
    last_op = last_op.fillna(lag["lag_168"])

    past = p.shift(HORIZON, freq="h").reindex(full_idx)          # T-168 까지의 전력
    temp = exog["기온"].reindex(full_idx).shift(HORIZON, freq="h").reindex(full_idx)
    out = pd.DataFrame({
        **lag,
        "same_hour_3w_mean": (lag["lag_168"] + lag["lag_336"] + lag["lag_504"]) / 3,
        "same_hour_3w_max": np.maximum.reduce([lag["lag_168"], lag["lag_336"], lag["lag_504"]]),
        "last_op_day_same_hour": last_op,
        "past_week_mean": past.rolling(168, min_periods=120).mean(),  # T-335 ~ T-168
        "past_week_max": past.rolling(168, min_periods=120).max(),
        "past_day_max_w": past.rolling(24, min_periods=18).max(),     # 1주 전 그날(같은 24시간) 최대
        "temp_week_mean": temp.rolling(168, min_periods=120).mean(),
        "hour": full_idx.hour,
        "day_of_week": full_idx.dayofweek,
    }, index=full_idx)
    return out.loc[index]


def plan_features(plan: pd.DataFrame) -> pd.DataFrame:
    """시간별 계획(on, production)으로 만든 계획 feature. 10_plan_regression.plan_features와 같은 규칙 + 생산량."""
    on = plan["on"].astype(int)
    prod = plan["production"].astype(float).where(on == 1, 0.0)
    idx = on.index
    date = idx.normalize()
    hour = pd.Series(idx.hour, index=idx)
    by_day = on.groupby(date)
    run = (on == 0).groupby(date).cumsum()
    rev = on[::-1]
    rev_run = (rev == 0).groupby(rev.index.normalize()).cumsum()
    on_hours = by_day.transform("sum")
    start = (by_day.transform("first") == 0) & (on_hours > 0)
    end = (by_day.transform("last") == 0) & (on_hours > 0)
    day_type = np.select([on_hours == 0, start & end, start, end, on_hours < 24],
                         ["비가동", "부분 가동", "가동 시작일", "가동 종료일", "중간 정지"], default="종일 가동")
    prod_day = prod.groupby(date).transform("sum")
    return pd.DataFrame({
        "plan_on": on,
        "plan_on_prev": on.shift(1).fillna(on).astype(int),
        "plan_on_next": on.shift(-1).fillna(on).astype(int),
        "plan_hours_since_on": on.groupby([date, run]).cumsum(),
        "plan_hours_until_off": rev.groupby([rev.index.normalize(), rev_run]).cumsum()[::-1],
        "plan_on_hours": on_hours,
        "plan_first_on_hour": hour.where(on == 1).groupby(date).transform("min").fillna(-1),
        "plan_last_on_hour": hour.where(on == 1).groupby(date).transform("max").fillna(-1),
        "plan_day_type": pd.Series(day_type, index=idx).map({t: i for i, t in enumerate(DAY_TYPES)}),
        "plan_prod": prod,
        "plan_prod_roll3": prod.rolling(3, center=True, min_periods=1).mean(),
        "plan_prod_day": prod_day,
    }, index=idx)


PROD_COLS = ["plan_prod", "plan_prod_roll3", "plan_prod_day"]


# =============================================================================
# 평소 일정(템플릿) 계획
# =============================================================================
def weekday_share(plan_hist: pd.DataFrame, copied: dict, cut: pd.Timestamp) -> pd.DataFrame:
    """cut 이전 원본 일자(복제일·회사 휴무 제외)의 요일·시각별 가동 비율. index = 요일(0=월), columns = 시각."""
    d = plan_hist.index.normalize()
    use = (plan_hist.index < cut) & ~d.isin(list(copied)) & ~d.isin(COMPANY_HOLIDAYS)
    part = plan_hist.loc[use, "on"]
    return part.groupby([part.index.dayofweek, part.index.hour]).mean().unstack().reindex(index=range(7), columns=range(24)).fillna(0.0)


def usual_plan(df: pd.DataFrame, plan_hist: pd.DataFrame, copied: dict, cut: pd.Timestamp,
               index: pd.DatetimeIndex) -> pd.DataFrame:
    """cut 이전 원본 일자로 만든 '평소 일정': 요일·시각별 가동 비율 > 0.5 이면 가동,
    생산량은 최근 TEMPLATE_WEEKS주 같은 요일·시각 평균(가동 시간만). 회사 휴무는 종일 비가동."""
    d = plan_hist.index.normalize()
    use = (plan_hist.index < cut) & ~d.isin(list(copied)) & ~d.isin(COMPANY_HOLIDAYS)
    part = plan_hist[use]
    share = weekday_share(plan_hist, copied, cut)
    recent = part[part.index >= cut - pd.Timedelta(weeks=TEMPLATE_WEEKS)]
    recent = recent[recent["on"] == 1]
    prod_mean = recent["production"].groupby([recent.index.dayofweek, recent.index.hour]).mean().unstack()
    prod_mean = prod_mean.reindex(index=range(7), columns=range(24)).fillna(0.0)

    plan_date = pd.DatetimeIndex([copied.get(x, x) for x in index.normalize()])
    on = (share.to_numpy()[plan_date.dayofweek, index.hour] > PATTERN_MIN_SHARE).astype(int)
    on[plan_date.isin(COMPANY_HOLIDAYS)] = 0
    prod = prod_mean.to_numpy()[plan_date.dayofweek, index.hour] * on
    return pd.DataFrame({"on": on, "production": np.round(prod, 0)}, index=index)


# =============================================================================
# 계획 CSV 입출력
# =============================================================================
def read_plan_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    need = {"date", "hour", "on"}
    if not need.issubset(raw.columns):
        raise ValueError(f"계획 CSV에 {sorted(need)} 열이 있어야 합니다. 현재 열: {list(raw.columns)}")
    idx = pd.to_datetime(raw["date"].astype(str)) + pd.to_timedelta(raw["hour"].astype(int), unit="h")
    if raw["on"].isna().any():
        raise ValueError("on 열에 빈 칸이 있습니다.")
    prod = (pd.to_numeric(raw["production"], errors="coerce").to_numpy(float)
            if "production" in raw else np.full(len(raw), np.nan))
    plan = pd.DataFrame({"on": raw["on"].astype(int).to_numpy(), "production": prod}, index=pd.DatetimeIndex(idx))
    if plan.index.duplicated().any():
        raise ValueError("같은 날짜·시간이 두 번 들어 있습니다.")
    plan = plan.sort_index()
    full = pd.date_range(plan.index.min(), plan.index.max(), freq="h")
    if len(full) != len(plan):
        raise ValueError(f"빠진 시간이 있습니다: {len(full) - len(plan)}개")
    if not set(plan["on"].unique()) <= {0, 1}:
        raise ValueError("on 열은 0 또는 1이어야 합니다.")
    return plan


def write_plan_csv(plan: pd.DataFrame, path: str, extra: pd.DataFrame | None = None) -> None:
    out = pd.DataFrame({"date": plan.index.strftime("%Y-%m-%d"), "hour": plan.index.hour,
                        "dow": [DOW_KO[d] for d in plan.index.dayofweek],
                        "on": plan["on"].astype(int).to_numpy(),
                        "production": plan["production"].round(0).astype(int).to_numpy()})
    if extra is not None:
        out = pd.concat([out, extra.reset_index(drop=True)], axis=1)
    out.to_csv(path, index=False, encoding="utf-8-sig")


def load_meta(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
