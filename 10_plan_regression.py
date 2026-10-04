"""
10 가동 정보 연계 전력 예측 (에이전트 M3): 하루 전에 내일의 시간별 전력(power_T)을 예측

06 회귀와의 차이
    - 예측 시점: 06은 1시간 전(직전 1시간 전력 사용), 10은 하루 전.
      전력·외생변수는 T보다 24시간 이상 앞선 값만 쓴다(lag_1~lag_12, 15분 값, 당일 rolling 제외).
    - 가동 계획: M1(08_operating_calendar.py)의 날 유형과 T 시각 가동 여부를 입력에 넣는다.

계획 정보 (보고서에 반드시 명시)
    - 현실 계획(B'): 하루 전에 실제로 알 수 있는 정보로만 만든다.
        1) 요일별 평소 일정: 학습 기간(검증 시작 전) 원본 일자에서 요일·시각별 가동 비율 > 0.5 이면 가동.
           복제일과 회사 휴무일은 패턴을 뽑을 때 뺀다. CV fold마다 그 fold 검증 시작 전 데이터로만 뽑는다.
        2) 회사 휴무 계획: 미리 공지되는 휴무(1/5 연초 휴무, 8/2~8/8 하계 휴무)는 종일 비가동.
           공휴일은 넣지 않는다(원본 공휴일 4일 모두 가동).
        3) 복제일은 원본 날짜의 계획을 물려받는다(전력 패턴이 원본 날짜를 따르므로).
      B'는 학습 기간에도 현실 계획을 넣고, B''는 학습은 실제 가동 기록으로 하고 예측할 때만 현실 계획을 넣는다.
      운영에서도 지난 기간은 실제 가동 기록이 있고 내일만 계획으로 알므로 B''가 실제 상황과 같다.
    - 실제 가동(B): 전력 > 시간 단위 가동 기준(M1, 41 kW)을 계획으로 간주. 예측 대상에서 만든 값이라
      "계획이 100% 지켜질 때"의 상한으로만 쓴다.
    - 생산 계획(C): 생산량 기록(전력과 독립적인 기록). 비교용.

실험 (XGBoost·LightGBM 각각, 7·8월 CV MAE로 선택, test는 1회 평가)
    A   기본                         : 하루 전 feature만 (가동 정보를 뺀 같은 모델)
    B'' 실제 가동으로 학습, 현실 계획으로 예측 : M3 본 모델
    B'  현실 계획으로 학습·예측       : 학습 기간 계획이 자주 어긋날 때의 비교
    B   실제 가동으로 학습·예측       : 계획이 그대로 지켜질 때의 상한
    C   기본 + 생산 계획              : 독립 기록만 쓴 경우

    CV 지표는 early stopping 때문에 약간 낙관적이다: 각 fold에서 early stopping 기준으로 검증 구간을 그대로 쓴다.
    test는 early stopping에 쓰지 않으므로 영향이 없다(06_regression.py와 같은 방식).

실행:  python preprocessing.py -> python 08_operating_calendar.py -> python 06_regression.py -> python 10_plan_regression.py
산출:  results/3_comparison/plan_regression_comparison.csv
       results/3_comparison/plan_regression_fold_comparison.csv  (7월·8월 fold별 CV 성능)
       results/2_test_predictions/plan_regression_predictions.csv, plan_regression_daily_max.csv  (M4·M5 입력)
       results/4_model_details/plan_regression_<실험>_<모델>_cv.csv
       results/4_model_details/plan_regression_oof.csv  (7·8월 CV 검증 구간 시간별 예측)
       results/4_model_details/plan_regression_weekday_pattern.csv  (현실 계획의 요일·시각별 가동 비율)
       results/3_comparison/plan_regression_plan_accuracy.csv  (현실 계획과 실제 가동의 일치율)
       results/5_models/plan_regression.joblib + _meta.json
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

import common as cm
import preprocessing as pp

warnings.filterwarnings("ignore", category=UserWarning)

HORIZON = 24                     # 하루 전: T - 24시간 이전 값만 사용
DAY_LAGS = [24, 48, 167, 168, 169, 336]
DAY_TYPES = ["비가동", "종일 가동", "가동 시작일", "가동 종료일", "중간 정지", "부분 가동"]
MAX_ROUNDS, EARLY_STOP = 3000, 100
# 미리 공지되는 회사 휴무: 원본 평일 비가동은 1/5(신정 연휴 다음 날)와 하계 휴무뿐이다.
# 하계 휴무는 주말까지 한 주(8/2~8/8)로 둔다. 실제로 8/7(토)도 비가동이었다.
COMPANY_HOLIDAYS = pd.DatetimeIndex(["2021-01-05"]).append(pd.date_range("2021-08-02", "2021-08-08"))
PATTERN_MIN_SHARE = 0.5          # 요일·시각별 가동 비율이 이 값을 넘으면 평소 일정상 가동
REAL_PLAN = "B'_현실계획"
PLAN_INFER = "B''_계획예측"     # 학습은 실제 가동, 예측할 때만 현실 계획
PLAN_ASSUMPTION = ("본 모델(B'')은 실제 가동 기록(전력 > M1 시간 단위 가동 기준)으로 학습하고, 예측할 때는 "
                   "현실 계획(학습 기간 요일별 평소 일정 + 미리 공지된 회사 휴무 1/5, 8/2~8/8)을 넣는다. "
                   "예측 시점에는 하루 전에 알 수 있는 정보만 쓴다. B는 예측에도 실제 가동을 넣은 상한이다.")


# =============================================================================
# feature 구성 (행 index = 예측 대상 시점 T)
# =============================================================================
def load_calendar() -> tuple[pd.DataFrame, float]:
    """M1 결과: 날짜별 가동 캘린더와 시간 단위 가동 기준."""
    cal = pd.read_csv(pp.out("prep", "operating_calendar.csv"), index_col="date", parse_dates=True, encoding="utf-8-sig")
    summary = pd.read_csv(pp.out("cmp", "operating_calendar_summary.csv"), index_col=0, encoding="utf-8-sig")
    return cal, float(summary.loc["hour_threshold", "전체"])


def base_features(df: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """하루 전에 알 수 있는 값: T-24 이전 전력·외생변수와 달력."""
    p, idx = df["power"], df.index
    date = idx.normalize()
    lag = {f"lag_{k}": p.shift(k) for k in DAY_LAGS}

    # 직전 "가동일"의 같은 시각 전력. 전날은 T-24 이후 값이 섞이므로 2~7일 전에서 찾는다.
    op_day = cal["operating"] == 1
    last_op_same_hour = pd.Series(np.nan, index=idx)
    for k in range(7, 1, -1):  # 먼 날부터 채우고 가까운 날로 덮어쓴다
        ok = pd.Series(op_day.reindex(date - pd.Timedelta(days=k)).to_numpy(), index=idx).fillna(False).astype(bool)
        last_op_same_hour = last_op_same_hour.mask(ok, p.shift(24 * k))
    last_op_same_hour = last_op_same_hour.fillna(lag["lag_168"])

    past = p.shift(HORIZON)  # T-24 까지의 전력
    return pd.concat(
        [
            pd.DataFrame(lag),
            pd.DataFrame({
                "same_hour_2w_mean": (lag["lag_168"] + lag["lag_336"]) / 2,
                "same_hour_2w_max": np.maximum(lag["lag_168"], lag["lag_336"]),
                "last_op_day_same_hour": last_op_same_hour,
                "past_day_mean": past.rolling(24).mean(),   # T-47 ~ T-24
                "past_day_max": past.rolling(24).max(),
                "past_day_off": (past.rolling(24).max() < pp.OFF_DAY_MAX_POWER).astype(int),
            }, index=idx),
            df[list(pp.EXOG_RENAME)].shift(HORIZON).rename(columns={k: f"{v}_lag24" for k, v in pp.EXOG_RENAME.items()}),
            pd.DataFrame({"hour": idx.hour, "day_of_week": idx.dayofweek}, index=idx),
        ],
        axis=1,
    )


def plan_features(on: pd.Series, prefix: str) -> pd.DataFrame:
    """시간별 계획 가동 여부(0/1)로 계획 feature 9개를 만든다. 날 유형은 M1(08)과 같은 규칙으로 판정한다."""
    idx = on.index
    date = idx.normalize()
    hour = pd.Series(idx.hour, index=idx)
    by_day = on.groupby(date)
    run = (on == 0).groupby(date).cumsum()                     # 같은 날 안에서 꺼질 때마다 새 구간
    rev = on[::-1]
    rev_run = (rev == 0).groupby(rev.index.normalize()).cumsum()
    on_hours = by_day.transform("sum")
    start = (by_day.transform("first") == 0) & (on_hours > 0)  # 0시 꺼짐 -> 이후 켜짐
    end = (by_day.transform("last") == 0) & (on_hours > 0)     # 23시 꺼짐
    day_type = np.select([on_hours == 0, start & end, start, end, on_hours < 24],
                         ["비가동", "부분 가동", "가동 시작일", "가동 종료일", "중간 정지"], default="종일 가동")
    return pd.DataFrame({
        "on": on,
        "on_prev": on.shift(1).fillna(on).astype(int),          # 전날 계획도 이미 확정된 값
        "on_next": by_day.shift(-1).fillna(on).astype(int),
        "hours_since_on": on.groupby([date, run]).cumsum(),
        "hours_until_off": rev.groupby([rev.index.normalize(), rev_run]).cumsum()[::-1],
        "on_hours": on_hours,
        "first_on_hour": hour.where(on == 1).groupby(date).transform("min").fillna(-1),
        "last_on_hour": hour.where(on == 1).groupby(date).transform("max").fillna(-1),
        "day_type": pd.Series(day_type, index=idx).map({t: i for i, t in enumerate(DAY_TYPES)}),
    }, index=idx).add_prefix(prefix)


def load_copied() -> dict:
    """복제일 -> 원본 날짜 (preprocessing.py가 저장)."""
    with open(pp.out("prep", "copied_days.json"), encoding="utf-8") as f:
        return {pd.Timestamp(k): pd.Timestamp(v) for k, v in json.load(f).items()}


def weekday_pattern(on: pd.Series, copied: dict, cut: pd.Timestamp) -> pd.DataFrame:
    """요일·시각별 가동 비율 (cut 이전 원본 일자, 회사 휴무일 제외). index = 요일(0=월), columns = 시각."""
    date = on.index.normalize()
    use = (on.index < cut) & ~date.isin(list(copied)) & ~date.isin(COMPANY_HOLIDAYS)
    part = on[use]
    return part.groupby([part.index.dayofweek, part.index.hour]).mean().unstack()


def realistic_plan(index: pd.DatetimeIndex, share: pd.DataFrame, copied: dict) -> pd.Series:
    """하루 전에 알 수 있는 계획: 요일별 평소 일정 + 회사 휴무. 복제일은 원본 날짜의 계획을 쓴다."""
    plan_date = pd.DatetimeIndex([copied.get(d, d) for d in index.normalize()])
    on = (share.to_numpy()[plan_date.dayofweek, index.hour] > PATTERN_MIN_SHARE).astype(int)
    on[plan_date.isin(COMPANY_HOLIDAYS)] = 0
    return pd.Series(on, index=index)


def production_features(df: pd.DataFrame) -> pd.DataFrame:
    """생산 계획: T 시각 생산량 기록 (전력과 독립적인 기록)."""
    prod_on = (df["생산량"] > 0).astype(int)
    return pd.DataFrame({
        "prod_plan": df["생산량"],
        "prod_plan_on": prod_on,
        "prod_plan_hours": prod_on.groupby(df.index.normalize()).transform("sum"),
    }, index=df.index)


def leakage_check(df: pd.DataFrame, cal: pd.DataFrame, base: pd.DataFrame) -> None:
    """특정 시각 이후의 전력·외생변수를 바꿔도 그 뒤 24시간 안의 기본 feature는 변하지 않아야 한다."""
    assert np.array_equal(base["lag_24"].dropna().to_numpy(), df["power"].shift(HORIZON).dropna().to_numpy())
    cut = pp.CV_FOLDS[0][0]
    changed = df.copy()
    num = changed.select_dtypes("number").columns
    changed.loc[changed.index >= cut, num] = changed.loc[changed.index >= cut, num] * 2 + 1
    window = (base.index >= cut) & (base.index < cut + pd.Timedelta(hours=HORIZON))
    pd.testing.assert_frame_equal(base_features(changed, cal).loc[window], base.loc[window])


# =============================================================================
# 데이터
# =============================================================================
df = pp.clean(pp.load_raw(), log=lambda *_: None)
ds, _, peak_thr = pp.load_dataset(log=lambda *_: None)  # 06과 같은 행·label·target로 평가
cal, hour_thr = load_calendar()
copied = load_copied()
actual_on = (df["power"] > hour_thr).astype(int)

base, plan, prod = base_features(df, cal), plan_features(actual_on, "plan_"), production_features(df)
leakage_check(df, cal, base)

# 현실 계획은 검증 시작 시점(cut)마다 그 이전 데이터로만 만든다: CV fold 2개 + 최종(9/1)
CUTS = [start for start, _ in pp.CV_FOLDS] + [pp.TEST_START]
SHARES = {cut: weekday_pattern(actual_on, copied, cut) for cut in CUTS}
RPLAN = {cut: plan_features(realistic_plan(df.index, SHARES[cut], copied), "rplan_") for cut in CUTS}
changed_on = actual_on.where(actual_on.index < CUTS[0], 1 - actual_on)  # cut 이후 가동 여부를 뒤집어도
assert weekday_pattern(changed_on, copied, CUTS[0]).equals(SHARES[CUTS[0]])  # 패턴은 그대로여야 한다

BASE, PLAN, PROD = list(base.columns), list(plan.columns), list(prod.columns)
RPLAN_COLS = list(RPLAN[pp.TEST_START].columns)
FEATURE_SETS = {
    "A_기본": BASE,
    REAL_PLAN: BASE + RPLAN_COLS,
    PLAN_INFER: BASE + PLAN,
    "B_가동계획": BASE + PLAN,
    "C_생산계획": BASE + PROD,
}
data = pd.concat([base, plan, prod, RPLAN[pp.TEST_START]], axis=1).loc[ds.index].assign(
    target_power=ds["target_power"], label=ds["label"], is_copied_day=ds["is_copied_day"])
assert data[BASE + PLAN + PROD + RPLAN_COLS].isna().sum().sum() == 0


def with_plan(frame: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    """현실 계획 열을 cut 이전 데이터로 만든 버전으로 바꾼다."""
    return frame.assign(**RPLAN[cut].loc[frame.index])


def predict_with_plan(frame: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    """B'': 실제 가동 열(plan_*)을 cut 이전 데이터로 만든 현실 계획 값으로 바꾼다. 예측할 때만 쓴다."""
    return frame.assign(**RPLAN[cut].loc[frame.index].rename(columns=lambda c: c.replace("rplan_", "plan_", 1)))


train_df, test_df = cm.final_split(data)
folds = cm.cv_folds(data)
fold_rows = np.mean([len(tr) for _, tr, _ in folds])
print(f"[정보] 하루 전 feature {len(BASE)}개 + 계획 {len(PLAN)}개 / 생산 계획 {len(PROD)}개, "
      f"시간 단위 가동 기준 > {hour_thr:.0f} kW")
print(f"[정보] train {len(train_df)}행 / test {len(test_df)}행 (피크 {int(test_df['label'].sum())}건, "
      f"가동 시간 {int(test_df['plan_on'].sum())}시간)")


# =============================================================================
# 평가지표
# =============================================================================
def daily_max(frame: pd.DataFrame, pred) -> pd.DataFrame:
    g = pd.DataFrame({"actual": frame["target_power"].to_numpy(), "pred": np.asarray(pred)}, index=frame.index)
    by = g.groupby(g.index.normalize())
    return pd.DataFrame({
        "actual_max": by["actual"].max(), "pred_max": by["pred"].max(),
        "actual_max_hour": by["actual"].idxmax().dt.hour, "pred_max_hour": by["pred"].idxmax().dt.hour,
    }).rename_axis("date")


def report(frame: pd.DataFrame, pred) -> dict:
    y, pred = frame["target_power"].to_numpy(), np.asarray(pred, dtype=float)
    peak, on = frame["label"].to_numpy() == 1, frame["plan_on"].to_numpy() == 1
    switch = frame["plan_on"].to_numpy() != frame["plan_on_prev"].to_numpy()  # 가동 시작·종료 시각
    dm = daily_max(frame, pred)
    mse = mean_squared_error(y, pred)

    def part(mask):
        return mean_absolute_error(y[mask], pred[mask]) if mask.any() else np.nan

    return {
        "MAE": mean_absolute_error(y, pred), "RMSE": float(np.sqrt(mse)), "MSE": mse,
        "peak_hour_MAE": part(peak), "on_hour_MAE": part(on), "off_hour_MAE": part(~on),
        "switch_hour_MAE": part(switch), "switch_hours": int(switch.sum()),
        "daily_max_MAE": float((dm["pred_max"] - dm["actual_max"]).abs().mean()),
        "daily_max_bias": float((dm["pred_max"] - dm["actual_max"]).mean()),
    }


# =============================================================================
# 모델 (06과 같은 탐색 범위)
# =============================================================================
MODELS = {
    "xgboost": {
        "grid": [{"max_depth": d, "learning_rate": lr, "min_child_weight": mcw, "objective": obj}
                 for d in [3, 5, 7] for lr in [0.03, 0.1] for mcw in [1, 5]
                 for obj in ["reg:squarederror", "reg:absoluteerror"]],
        "make": lambda p, n, es=None: XGBRegressor(**p, n_estimators=n, subsample=0.8, colsample_bytree=0.8,
                                                   tree_method="hist", random_state=cm.SEED, n_jobs=-1,
                                                   early_stopping_rounds=es),
    },
    "lightgbm": {
        "grid": [{"num_leaves": nl, "learning_rate": lr, "min_child_samples": mcs, "objective": obj}
                 for nl in [7, 15, 31] for lr in [0.03, 0.1] for mcs in [10, 30]
                 for obj in ["regression", "regression_l1"]],
        "make": lambda p, n, es=None: LGBMRegressor(**p, n_estimators=n, subsample=0.8, subsample_freq=1,
                                                    colsample_bytree=0.8, random_state=cm.SEED, n_jobs=-1, verbose=-1),
    },
}


def fit_fold(key, params, feats, tr, va):
    spec = MODELS[key]
    if key == "xgboost":
        m = spec["make"](params, MAX_ROUNDS, EARLY_STOP)
        m.fit(tr[feats], tr["target_power"], eval_set=[(va[feats], va["target_power"])], verbose=False)
        return m.predict(va[feats]), m.best_iteration + 1
    m = spec["make"](params, MAX_ROUNDS)
    m.fit(tr[feats], tr["target_power"], eval_set=[(va[feats], va["target_power"])],
          eval_metric="l1", callbacks=[early_stopping(EARLY_STOP, verbose=False)])
    return m.predict(va[feats]), m.best_iteration_


def run(set_name: str, key: str) -> dict:
    """한 feature 묶음·모델을 CV로 튜닝하고 9/1 이전 전체로 재학습해 test를 예측한다."""
    spec, feats = MODELS[key], FEATURE_SETS[set_name]
    set_folds = folds
    if set_name == REAL_PLAN:  # fold마다 그 fold 검증 시작 전 데이터로 만든 현실 계획을 쓴다
        set_folds = [(n, with_plan(tr, cut), with_plan(va, cut)) for (n, tr, va), cut in zip(folds, CUTS)]
    elif set_name == PLAN_INFER:  # 과거는 실제 가동 기록으로 학습하고, 검증 구간은 계획만 안다
        set_folds = [(n, tr, predict_with_plan(va, cut)) for (n, tr, va), cut in zip(folds, CUTS)]
    cv_rows = []
    for params in spec["grid"]:
        preds, its = [], []
        for _, tr, va in set_folds:
            pred, it = fit_fold(key, params, feats, tr, va)
            preds.append(pred); its.append(it)
        fold_mae = [mean_absolute_error(va["target_power"], pr) for (_, _, va), pr in zip(folds, preds)]
        cv_rows.append({**params, "cv_MAE": mean_absolute_error(oof_frame["target_power"], np.concatenate(preds)),
                        "fold_MAE": " / ".join(f"{v:.2f}" for v in fold_mae),
                        "best_iter": int(np.mean(its)), "_oof": np.concatenate(preds)})
    cv = pd.DataFrame(cv_rows).sort_values("cv_MAE")
    cv.drop(columns="_oof").to_csv(cm.out("detail", f"plan_regression_{set_name.replace(chr(39), 'p')}_{key}_cv.csv"),
                                   index=False)
    best = cv.iloc[0]
    params = {k: best[k] for k in spec["grid"][0]}
    params = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
              for k, v in params.items()}
    n_final = int(round(best["best_iter"] * len(train_df) / fold_rows))

    model = spec["make"](params, n_final)
    model.fit(train_df[feats], train_df["target_power"])
    test_x = predict_with_plan(test_df, pp.TEST_START) if set_name == PLAN_INFER else test_df
    pred_test = model.predict(test_x[feats])
    cv_m, test_m = report(oof_frame, best["_oof"]), report(test_df, pred_test)
    print(f"[{set_name} / {key}] {params}, n={n_final} | CV MAE {cv_m['MAE']:.2f} (fold {best['fold_MAE']}) | "
          f"test MAE {test_m['MAE']:.2f}, 피크 시간 {test_m['peak_hour_MAE']:.2f}, 가동 시간 {test_m['on_hour_MAE']:.2f}, "
          f"일 최대 {test_m['daily_max_MAE']:.2f}")
    return {"set": set_name, "model": key, "features": feats, "params": {**params, "n_estimators": n_final},
            "fitted": model, "pred_test": pred_test, "pred_oof": best["_oof"], "cv": cv_m, "test": test_m}


oof_frame = pd.concat([va for _, _, va in folds])
results = [run(s, k) for s in FEATURE_SETS for k in MODELS]

# =============================================================================
# 비교표: 실험 6개 + 기준 3개 (기준은 다시 학습하지 않는다)
# =============================================================================
rows = [{"Model": f"{r['set']} / {r['model']}", "시점": "하루 전", **r["test"],
         "cv_MAE": r["cv"]["MAE"], "cv_on_hour_MAE": r["cv"]["on_hour_MAE"], "cv_peak_hour_MAE": r["cv"]["peak_hour_MAE"],
         "n_features": len(r["features"]), "params": str(r["params"])} for r in results]
rows.append({"Model": "1주 전 같은 시각 (lag_168)", "시점": "하루 전", **report(test_df, test_df["lag_168"]),
             "cv_MAE": report(oof_frame, oof_frame["lag_168"])["MAE"]})
rows.append({"Model": "Persistence (lag_1)", "시점": "1시간 전", **report(test_df, ds.loc[test_df.index, "lag_1"])})
hour_ahead_path = cm.out("pred", "xgboost_reg_predictions.csv")
if os.path.exists(hour_ahead_path):
    hour_ahead = pd.read_csv(hour_ahead_path, index_col="Date", parse_dates=True)["predicted_power"]
    rows.append({"Model": "06 xgboost_reg (직전 값 포함)", "시점": "1시간 전", **report(test_df, hour_ahead.loc[test_df.index])})
else:
    print("[주의] 06_regression.py 결과가 없어 1시간 전 모델 비교를 건너뜀")
table = pd.DataFrame(rows).set_index("Model")
table.to_csv(cm.out("cmp", "plan_regression_comparison.csv"), encoding="utf-8-sig")

# fold별 CV 성능: 7월(평소 가동)과 8월(하계 휴무 포함)을 나눠 본다. 각 fold는 검증 시작 전 데이터로만 학습한 예측이다.
per_fold, start = [], 0
for name, _, va in folds:
    part = slice(start, start + len(va))
    for r in results:
        per_fold.append({"Model": f"{r['set']} / {r['model']}", "fold": name, "hours": len(va),
                          **report(va, r["pred_oof"][part])})
    per_fold.append({"Model": "1주 전 같은 시각 (lag_168)", "fold": name, "hours": len(va), **report(va, va["lag_168"])})
    start += len(va)
fold_table = pd.DataFrame(per_fold).set_index(["Model", "fold"])
fold_table.to_csv(cm.out("cmp", "plan_regression_fold_comparison.csv"), encoding="utf-8-sig")

# M3 본 모델 = 예측 시점에 현실 계획만 쓰는 실험(B', B'') 중 CV MAE가 가장 낮은 모델.
# 같은 알고리즘의 다른 실험을 함께 저장해 비교한다.
by_name = {(r["set"], r["model"]): r for r in results}
m3 = min((r for r in results if r["set"] in (REAL_PLAN, PLAN_INFER)), key=lambda r: r["cv"]["MAE"])
no_plan, full_plan, prod_plan = (by_name[(s, m3["model"])] for s in ["A_기본", "B_가동계획", "C_생산계획"])
other_real = by_name[(REAL_PLAN if m3["set"] == PLAN_INFER else PLAN_INFER, m3["model"])]

# 현실 계획이 실제 가동과 얼마나 맞는지 (CV fold는 그 fold의 계획, test는 9/1 기준 계획)
periods = [(name, va, cut) for (name, _, va), cut in zip(folds, CUTS)] + [("test", test_df, pp.TEST_START)]
oof_rplan_on = pd.concat([RPLAN[cut].loc[va.index, "rplan_on"] for _, va, cut in periods[:-1]])
acc_rows = []
for name, frame, cut in periods:
    planned, actual = RPLAN[cut].loc[frame.index, "rplan_on"], frame["plan_on"]
    miss = planned != actual
    miss_days = sorted({d.strftime("%m/%d") for d in frame.index[miss.to_numpy()].normalize()})
    acc_rows.append({"period": name, "hours": len(frame), "agreement": float((~miss).mean()),
                     "planned_on_actual_off": int(((planned == 1) & (actual == 0)).sum()),
                     "planned_off_actual_on": int(((planned == 0) & (actual == 1)).sum()),
                     "mismatch_days": len(miss_days), "mismatch_dates": " ".join(miss_days)})
plan_acc = pd.DataFrame(acc_rows).set_index("period")
plan_acc.to_csv(cm.out("cmp", "plan_regression_plan_accuracy.csv"), encoding="utf-8-sig")
SHARES[pp.TEST_START].rename(index=dict(enumerate(["월", "화", "수", "목", "금", "토", "일"]))).rename_axis("dow").to_csv(
    cm.out("detail", "plan_regression_weekday_pattern.csv"), encoding="utf-8-sig")


def prediction_table(frame: pd.DataFrame, key: str, planned_on) -> pd.DataFrame:
    """시간별 실제·예측 전력. key = "pred_test"(test) 또는 "pred_oof"(7·8월 CV 검증 구간)."""
    return pd.DataFrame({
        "Date": frame.index,
        "actual_power": frame["target_power"].to_numpy(),
        "predicted_power": m3[key],                       # 본 모델 (예측 시점에 현실 계획만 사용)
        f"predicted_power_{other_real['set'].split('_')[0].replace(chr(39), 'p')}": other_real[key],
        "predicted_power_full_plan": full_plan[key],      # B 실제 가동 = 계획이 그대로 지켜질 때
        "predicted_power_no_plan": no_plan[key],          # A
        "predicted_power_prod_plan": prod_plan[key],      # C
        "planned_on": np.asarray(planned_on),             # 현실 계획상 가동 여부
        "plan_on": frame["plan_on"].to_numpy(),           # 실제 가동 여부
        "day_type": cal["day_type"].reindex(frame.index.normalize()).to_numpy(),
        "actual_label": frame["label"].to_numpy(),
    })


prediction_table(test_df, "pred_test", test_df["rplan_on"]).to_csv(
    cm.out("pred", "plan_regression_predictions.csv"), index=False, encoding="utf-8-sig")
# 7·8월 CV out-of-fold 예측 (복제일 없는 구간). M4에서 test를 보지 않고 조정 규칙을 고를 때 쓴다.
prediction_table(oof_frame, "pred_oof", oof_rplan_on).to_csv(
    cm.out("detail", "plan_regression_oof.csv"), index=False, encoding="utf-8-sig")
dm = daily_max(test_df, m3["pred_test"])
dm.assign(error=dm["pred_max"] - dm["actual_max"]).to_csv(cm.out("pred", "plan_regression_daily_max.csv"), encoding="utf-8-sig")

cm.save_model("plan_regression", m3["fitted"], {
    "script": "10_plan_regression.py", "task": "하루 전 시간별 전력 회귀 (가동 계획 연계)",
    "target": "target_power (연속값)", "forecast_horizon": f"T-{HORIZON}시간 이전 값만 사용",
    "plan_assumption": PLAN_ASSUMPTION, "hour_on_threshold": hour_thr, "day_types": DAY_TYPES,
    "company_holidays": [d.strftime("%Y-%m-%d") for d in COMPANY_HOLIDAYS], "pattern_min_share": PATTERN_MIN_SHARE,
    "weekday_pattern_file": "4_model_details/plan_regression_weekday_pattern.csv (9/1 이전 데이터로 만든 계획)",
    "algorithm": m3["model"], "features": m3["features"], "params": m3["params"],
    "cv": {k: round(float(v), 4) for k, v in m3["cv"].items()},
    "test": {k: round(float(v), 4) for k, v in m3["test"].items()},
})

cols = ["시점", "MAE", "RMSE", "peak_hour_MAE", "on_hour_MAE", "switch_hour_MAE", "daily_max_MAE", "cv_MAE"]
print("\n=== fold별 CV 성능 (7월 / 8월) ===")
print(fold_table[["hours", "MAE", "RMSE", "peak_hour_MAE", "on_hour_MAE", "daily_max_MAE"]].round(2).unstack("fold").to_string())
print("\n=== Test 회귀 성능 (하루 전 vs 1시간 전) ===")
print(table[cols].round(2).to_string())
print("\n=== 현실 계획과 실제 가동의 일치 ===")
print(plan_acc.round(4).to_string())
pattern = SHARES[pp.TEST_START] > PATTERN_MIN_SHARE
for d, name in enumerate(["월", "화", "수", "목", "금", "토", "일"]):
    hours = np.flatnonzero(pattern.loc[d].to_numpy())
    print(f"  {name}: " + (f"{hours.min()}~{hours.max()}시 가동 ({len(hours)}시간)" if len(hours) else "비가동"))
gain = no_plan["test"]["MAE"] - m3["test"]["MAE"]
print(f"\n[M3 본 모델] {m3['set']} / {m3['model']}")
print(f"  현실 계획 효과: MAE {no_plan['test']['MAE']:.2f} -> {m3['test']['MAE']:.2f} ({gain:+.2f} kW 개선), "
      f"CV {no_plan['cv']['MAE']:.2f} -> {m3['cv']['MAE']:.2f}")
print(f"  {other_real['set']}: test {other_real['test']['MAE']:.2f}, CV {other_real['cv']['MAE']:.2f}")
print(f"  계획이 그대로 지켜질 때(B): test {full_plan['test']['MAE']:.2f}, CV {full_plan['cv']['MAE']:.2f}")
print(f"  생산 계획만 쓴 경우: MAE {prod_plan['test']['MAE']:.2f}")
print(f"  가이드북 비교용 MSE: {m3['test']['MSE']:.2f} (= RMSE², 예측 대상 정의가 같은지 확인 후 사용)")
print(f"[가정] {PLAN_ASSUMPTION}")
