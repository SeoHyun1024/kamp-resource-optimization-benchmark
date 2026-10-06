"""
15 주간 피드백 (관리자 화면용): 매주 '그 전주까지의 데이터'로 다시 학습해 그 주를 예측·조정하고 실제와 비교

목적
    실제 운영처럼 주가 지날 때마다 모델을 갱신했다면 일정 조정(M4)이 얼마나 효과가 있었는지,
    그리고 예측이 완벽했다면 어디까지 가능했는지를 주마다 나란히 보여준다.

대상 주: 2021-07-07(수) ~ 2021-09-14(화), 수요일 시작 10주 (test 9/1 시작과 같은 요일)
    1~6월은 다른 날짜를 복사한 '복제일'이 많아 실제와 비교하는 피드백이 왜곡되므로 7월부터 본다.
    7/30은 복제일이라 지표 계산에서 뺀다.

주마다 하는 일 (주 시작일 s)
    1) M3 하루 전 예측: 10_plan_regression.py 본 모델(B'' / LightGBM)과 같은 feature·하이퍼파라미터로
       s 이전 데이터만 써서 다시 학습한다. 트리 수는 학습 행 수에 비례해 맞춘다(10과 같은 규칙).
       예측할 때 가동 계획은 s 이전 데이터로 만든 '현실 계획'(요일별 평소 일정 + 회사 휴무)을 쓴다.
       -> 9/1 주는 10의 test 예측과 완전히 같아야 한다 (assert로 확인).
    2) M2 피크 경보: RNN·RF를 주마다 다시 학습하면 몇 시간이 걸려, 이미 있는 '월 단위' 결과를 쓴다.
       7·8월 = CV OOF(그 달 이전 데이터로 학습), 9월 = test 예측(8/31까지 학습). 규칙은 09와 같다
       (주의 = RF 확률 >= 0.55, 확정 = 주의 AND RNN 예측 >= 169). 9월은 09 결과와 같은지 확인한다.
    3) M4 일정 조정: 11_schedule_adjust.py 의 adjust_day 를 그대로 쓴다.
         AI 계획  : 1)의 예측과 현실 계획으로 조정안을 짜고 실제 전력에 적용 (미래 정보 없음)
         사후 최선: 같은 규칙에 '실제 전력'을 예측 대신 넣고(완벽한 예측, 마진 0) 실제 가동 기록으로 조정
                    = 그 주를 다 알았다면 이 규칙으로 갈 수 있는 상한. AI 계획과의 차이 = 예측 오차로 놓친 몫
    4) 피드백 문구: 예측 편향이 큰 요일, 놓친 피크 시간, 계획(평소 일정)과 실제 가동이 다른 날 등

실행: python 15_weekly_feedback.py   (08·09·10 결과 필요, 약 1분)
산출: results/4_model_details/weekly_feedback.json        (12_build_report.py -> 웹 관리자 화면)
      results/3_comparison/weekly_feedback_summary.csv     (주별 요약, 기본 시나리오)
      results/2_test_predictions/weekly_feedback_hourly.csv (시간별 예측·경보·조정, 기본 시나리오)
"""
from __future__ import annotations

import ast
import importlib
import json
import os
import sys
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import preprocessing as pp

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
WEEK_STARTS = list(pd.date_range("2021-07-07", "2021-09-08", freq="7D"))
TEST_START = pp.TEST_START


# =============================================================================
# 10_plan_regression.py 의 함수를 그대로 가져온다 (파일 전체를 import 하면 50분짜리 학습이 돌기 때문에
# 함수·상수 정의만 골라 실행). 코드를 복사하지 않아 10과 feature가 어긋날 일이 없다.
# =============================================================================
def load_defs(path: str, names: set[str]) -> dict:
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            keep.append(node)
    ns = {"__name__": "m3_defs"}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    missing = names - set(ns)
    assert not missing, f"10_plan_regression.py 에서 찾지 못함: {missing}"
    return ns


M3 = load_defs(os.path.join(HERE, "10_plan_regression.py"),
               {"HORIZON", "DAY_LAGS", "DAY_TYPES", "COMPANY_HOLIDAYS", "PATTERN_MIN_SHARE",
                "load_calendar", "base_features", "plan_features", "load_copied", "weekday_pattern", "realistic_plan"})
M4 = importlib.import_module("11_schedule_adjust")


def load_json(*parts):
    with open(os.path.join(pp.RESULTS_DIR, *parts), encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# 1) M3 주별 재학습
# =============================================================================
def m3_walk_forward():
    df = pp.clean(pp.load_raw(), log=lambda *_: None)
    ds, _, peak_thr = pp.load_dataset(log=lambda *_: None)
    cal, hour_thr = M3["load_calendar"]()
    copied = M3["load_copied"]()
    actual_on = (df["power"] > hour_thr).astype(int)
    base = M3["base_features"](df, cal)
    plan = M3["plan_features"](actual_on, "plan_")
    data = pd.concat([base, plan], axis=1).loc[ds.index].assign(
        target_power=ds["target_power"], is_copied_day=ds["is_copied_day"])

    meta = load_json("5_models", "plan_regression_meta.json")
    assert meta["algorithm"] == "lightgbm"
    feats = meta["features"]
    params = {k: v for k, v in meta["params"].items() if k != "n_estimators"}
    n_ref = meta["params"]["n_estimators"]
    rows_ref = int((data.index < TEST_START).sum())       # 10의 최종 학습 행 수 (9/1 이전 전체)

    pred = pd.Series(np.nan, index=data.index)
    planned = pd.Series(np.nan, index=data.index)
    weeks = []
    for s in WEEK_STARTS:
        e = s + pd.Timedelta(days=7)
        tr = data[data.index < s]
        te = data[(data.index >= s) & (data.index < e)]
        n = max(1, int(round(n_ref * len(tr) / rows_ref)))
        model = LGBMRegressor(**params, n_estimators=n, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=pp.SEED, n_jobs=-1, verbose=-1)
        model.fit(tr[feats], tr["target_power"])
        share = M3["weekday_pattern"](actual_on, copied, s)
        rplan_on = M3["realistic_plan"](df.index, share, copied)
        rplan = M3["plan_features"](rplan_on, "plan_")       # 예측 시점에는 현실 계획만 안다 (B'')
        x = te.assign(**rplan.loc[te.index])
        pred.loc[te.index] = model.predict(x[feats])
        planned.loc[te.index] = rplan_on.loc[te.index]
        weeks.append({"start": s, "end": e - pd.Timedelta(days=1), "train_until": tr.index.max(),
                      "n_estimators": n, "train_rows": len(tr)})

    # 검증: 9/1 주는 10의 test 예측과 같아야 한다 (같은 학습 데이터·트리 수·feature)
    ref = pd.read_csv(pp.out("pred", "plan_regression_predictions.csv"), parse_dates=["Date"], index_col="Date")
    wk = ref.index[(ref.index >= TEST_START) & (ref.index < TEST_START + pd.Timedelta(days=7))]
    assert np.allclose(pred.loc[wk].to_numpy(), ref.loc[wk, "predicted_power"].to_numpy(), atol=1e-6), \
        "9/1 주 예측이 10_plan_regression.py 결과와 다릅니다"
    assert (planned.loc[wk].astype(int).to_numpy() == ref.loc[wk, "planned_on"].to_numpy()).all()
    return df, data, cal, actual_on, pred, planned, weeks, peak_thr


# =============================================================================
# 2) M2 경보 (월 단위 결과)
# =============================================================================
def m2_levels(index) -> tuple[pd.Series, pd.Series, pd.Series]:
    meta = load_json("5_models", "peak_alert_meta.json")
    assert meta["rules"] == {"주의": "RF 단독", "확정": "AND"}, "09의 규칙이 바뀌었습니다. 15의 경보 계산을 맞춰 주세요."
    rf_thr, rnn_cut = float(meta["rf_threshold"]), float(meta["rnn_cutoff"])
    rf_oof = pd.read_csv(pp.out("detail", "random_forest_oof.csv"), parse_dates=["Date"], index_col="Date")["score"]
    rnn_oof = pd.read_csv(pp.out("detail", "rnn_oof.csv"), parse_dates=["Date"], index_col="Date")["forecast"]
    rf_te = pd.read_csv(pp.out("pred", "random_forest_predictions.csv"), parse_dates=["Date"], index_col="Date")["risk_probability"]
    rnn_te = pd.read_csv(pp.out("pred", "rnn_forecast.csv"), parse_dates=["Date"], index_col="Date")["forecast"]
    rf = pd.concat([rf_oof, rf_te]).reindex(index)
    rnn = pd.concat([rnn_oof, rnn_te]).reindex(index)
    caution = rf >= rf_thr
    confirm = caution & (rnn >= rnn_cut)
    level = pd.Series(np.select([confirm, caution], ["확정", "주의"], "정상"), index=index)
    level[rf.isna()] = ""
    ref = pd.read_csv(pp.out("pred", "peak_alerts.csv"), parse_dates=["Date"], index_col="Date")["level"]
    common = ref.index.intersection(index)
    assert (level.loc[common] == ref.loc[common]).all(), "9월 경보가 09_peak_alert.py 결과와 다릅니다"
    return level, rf, rnn


# =============================================================================
# 3) M4 조정 + 4) 피드백
# =============================================================================
def build_days(idx, actual, pred, on, cal, m2):
    days: "OrderedDict[str, list]" = OrderedDict()
    for t in idx:
        d = f"{t:%Y-%m-%d}"
        days.setdefault(d, []).append({
            "h": t.hour, "actual": float(actual[t]), "pred": float(pred[t]), "on": int(on[t]),
            "day_type": str(cal["day_type"].get(t.normalize(), "")), "m2": m2.get(t, ""),
        })
    return days


def week_summary(day_list, days, adj, thr):
    """한 주의 실제 / 조정 후 전력 요약."""
    act = [v for d in day_list for v in (x["actual"] for x in days[d])]
    aft = [v for d in day_list for v in adj[d]["actual_after"]]
    return {"peak_hours": int(sum(v >= thr for v in aft)), "week_max": round(max(aft), 1),
            "peak_hours_actual": int(sum(v >= thr for v in act)), "week_max_actual": round(max(act), 1),
            "moved_kwh": round(sum(sum(adj[d]["out"]) for d in day_list), 1)}


DOW = ["월", "화", "수", "목", "금", "토", "일"]


def feedback_notes(day_list, days, thr, margin, copied_days):
    """다음 주에 반영할 점 (자동 문구, 최대 4개)."""
    notes = []
    rows = [(d, x) for d in day_list if d not in copied_days for x in days[d]]
    if not rows:
        return notes
    # (a) 일 최대 예측 편향이 가장 큰 날
    bias = []
    for d in day_list:
        if d in copied_days:
            continue
        a = max(x["actual"] for x in days[d])
        p = max(x["pred"] for x in days[d])
        if a >= pp.OFF_DAY_MAX_POWER:  # 가동일만
            bias.append((p - a, d))
    if bias:
        worst = min(bias)
        if worst[0] <= -8:
            d = pd.Timestamp(worst[1])
            notes.append(f"{d.month}/{d.day}({DOW[d.dayofweek]}) 일 최대를 {-worst[0]:.0f}kW 낮게 예측했어요. "
                         f"같은 요일·날 유형은 안전 마진을 더 두는 게 좋아요.")
        mean_b = float(np.mean([b for b, _ in bias]))   # 위 m3.daily_max_bias 와 같은 정의 (가동일)
        if mean_b <= -5:
            notes.append(f"이번 주 일 최대 예측이 평균 {-mean_b:.1f}kW 낮았어요 (기본 안전 마진 {margin:.0f}kW와 비교).")
    # (b) 피크를 놓친 시간대
    missed = [(d, x["h"]) for d, x in rows if x["actual"] >= thr and x["pred"] + margin < thr]
    if missed:
        hours = pd.Series([h for _, h in missed]).value_counts()
        top = hours.index[0]
        often = f" ({top:02d}시에 {hours.iloc[0]}번)" if hours.iloc[0] >= 2 else ""
        notes.append(f"실제 피크 {sum(1 for _, x in rows if x['actual'] >= thr)}시간 중 {len(missed)}시간은 예측+마진으로도 못 잡았어요"
                     f"{often}. 이런 시간은 당일 M2 경보로 대응해야 해요.")
    # (c) 계획(평소 일정)과 실제 가동이 다른 날
    diff_days = []
    for d in day_list:
        if d in copied_days:
            continue
        mism = sum(1 for x in days[d] if x["on"] != int(x["actual"] > 41))
        if mism >= 4:
            diff_days.append((d, mism))
    if diff_days:
        txt = ", ".join(f"{pd.Timestamp(d).month}/{pd.Timestamp(d).day}({m}시간)" for d, m in diff_days[:3])
        notes.append(f"평소 일정 계획과 실제 가동이 다른 날: {txt}. 이런 날은 실제 계획을 미리 넣으면 예측이 좋아져요.")
    return notes[:4]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    df, data, cal, actual_on, pred, planned, weeks, thr = m3_walk_forward()
    idx = data.index[(data.index >= WEEK_STARTS[0]) & (data.index < WEEK_STARTS[-1] + pd.Timedelta(days=7))]
    level, rf, rnn = m2_levels(idx)
    copied_days = {f"{pd.Timestamp(k):%Y-%m-%d}" for k in M3["load_copied"]()}
    actual = data.loc[idx, "target_power"]
    days_ai = build_days(idx, actual, pred.loc[idx], planned.loc[idx].astype(int), cal, level)
    days_hs = build_days(idx, actual, actual, actual_on.loc[idx], cal, level)   # 사후: 예측 = 실제, 계획 = 실제 가동

    scen = OrderedDict()
    for share in M4.SHARES:
        hs = OrderedDict((d, M4.adjust_day(h, thr, share, 0.0)) for d, h in days_hs.items())
        for margin in M4.MARGINS:
            ai = OrderedDict((d, M4.adjust_day(h, thr, share, margin)) for d, h in days_ai.items())
            scen[f"r{int(round(share * 100))}_m{int(margin)}"] = {"share": share, "margin": margin, "ai": ai, "hs": hs,
                                                                   "summary": M4.summarize(days_ai, ai, thr)}
    default = f"r{int(round(M4.DEFAULT_SHARE * 100))}_m{int(M4.DEFAULT_MARGIN)}"

    # 주별 지표
    out_weeks = []
    for w in weeks:
        dl = [f"{d:%Y-%m-%d}" for d in pd.date_range(w["start"], w["end"])]
        use = [d for d in dl if d not in copied_days]
        rows_t = [pd.Timestamp(d) + pd.Timedelta(hours=x["h"]) for d in use for x in days_ai[d]]
        a, p = actual.loc[rows_t].to_numpy(), pred.loc[rows_t].to_numpy()
        dmax = [(max(x["pred"] for x in days_ai[d]), max(x["actual"] for x in days_ai[d])) for d in use
                if max(x["actual"] for x in days_ai[d]) >= pp.OFF_DAY_MAX_POWER]   # 일 최대 지표는 가동일만
        lv = level.loc[rows_t]
        peaks = a >= thr
        per = {}
        for k, sc in scen.items():
            per[k] = {"ai": week_summary(use, days_ai, sc["ai"], thr), "hs": week_summary(use, days_hs, sc["hs"], thr)}
        out_weeks.append({
            "start": f"{w['start']:%Y-%m-%d}", "end": f"{w['end']:%Y-%m-%d}",
            "train_until": f"{w['train_until']:%Y-%m-%d}", "train_rows": w["train_rows"], "n_estimators": w["n_estimators"],
            "days": dl, "excluded_days": [d for d in dl if d in copied_days],
            "m3": {"mae": round(float(np.mean(np.abs(a - p))), 2),
                   "daily_max_mae": round(float(np.mean([abs(x - y) for x, y in dmax])), 2) if dmax else None,
                   "daily_max_bias": round(float(np.mean([x - y for x, y in dmax])), 2) if dmax else None},
            "m2": {"peaks": int(peaks.sum()), "caught": int((peaks & lv.isin(["주의", "확정"]).to_numpy()).sum()),
                   "alerts": int(lv.isin(["주의", "확정"]).sum()), "month_model": "월 단위 (그 달 이전 데이터로 학습)"},
            "scenarios": per,
            "notes": feedback_notes(dl, days_ai, thr, scen[default]["margin"], copied_days),
        })

    # 저장
    hourly = []
    sd = scen[default]
    for d, hours in days_ai.items():
        for x in hours:
            h = x["h"]
            hourly.append({"Date": f"{d} {h:02d}:00:00", "actual_power": x["actual"], "pred_walk_forward": round(x["pred"], 2),
                           "planned_on": x["on"], "m2_level": x["m2"],
                           "ai_actual_after": sd["ai"][d]["actual_after"][h], "hindsight_actual_after": sd["hs"][d]["actual_after"][h]})
    pd.DataFrame(hourly).to_csv(pp.out("pred", "weekly_feedback_hourly.csv"), index=False, encoding="utf-8-sig")
    summ = pd.DataFrame([{
        "week": f"{w['start']} ~ {w['end']}", "train_until": w["train_until"], "m3_mae": w["m3"]["mae"],
        "m3_daily_max_bias": w["m3"]["daily_max_bias"], "peaks_actual": w["scenarios"][default]["ai"]["peak_hours_actual"],
        "peaks_ai": w["scenarios"][default]["ai"]["peak_hours"], "peaks_hindsight": w["scenarios"][default]["hs"]["peak_hours"],
        "max_actual": w["scenarios"][default]["ai"]["week_max_actual"], "max_ai": w["scenarios"][default]["ai"]["week_max"],
        "max_hindsight": w["scenarios"][default]["hs"]["week_max"], "m2_caught": f"{w['m2']['caught']}/{w['m2']['peaks']}",
    } for w in out_weeks])
    summ.to_csv(pp.out("cmp", "weekly_feedback_summary.csv"), index=False, encoding="utf-8-sig")

    payload = {
        "meta": {"script": "15_weekly_feedback.py", "peak_threshold": thr, "default": default,
                 "shares": M4.SHARES, "margins": M4.MARGINS, "dest_buffer_kw": M4.DEST_BUFFER, "max_shift_h": M4.MAX_SHIFT_H,
                 "period": [days_ai and next(iter(days_ai)), list(days_ai)[-1]],
                 "m3_params": load_json("5_models", "plan_regression_meta.json")["params"],
                 "excluded_days": sorted(copied_days & set(days_ai))},
        "days": {d: {"actual": [x["actual"] for x in h], "pred": [round(x["pred"], 1) for x in h],
                     "on": [x["on"] for x in h], "on_actual": [int(x["actual"] > 41) for x in h],
                     "m2": [x["m2"] for x in h], "day_type": h[0]["day_type"]} for d, h in days_ai.items()},
        "scenarios": {k: {"share": s["share"], "margin": s["margin"], "summary": s["summary"], "days": s["ai"]}
                      for k, s in scen.items()},
        # 사후 최선은 마진과 무관(마진 0)이라 이동 가능 비율별로 한 번만 저장
        "hindsight": {f"r{int(round(s['share'] * 100))}": {d: {"actual_after": v["actual_after"], "out": v["out"], "in": v["in"]}
                                                           for d, v in s["hs"].items()} for s in scen.values()},
        "weeks": out_weeks,
    }
    with open(pp.out("detail", "weekly_feedback.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    pd.set_option("display.width", 200)
    print(f"[검증] 9/1 주 예측 = 10_plan_regression.py 결과, 9월 경보 = 09_peak_alert.py 결과 (일치)")
    print(f"[기본 시나리오 {default}] 주별 결과")
    print(summ.to_string(index=False))
    return payload


if __name__ == "__main__":
    main()
