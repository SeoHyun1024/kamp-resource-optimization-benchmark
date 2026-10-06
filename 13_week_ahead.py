"""
13 1주 전 계획 연계 전력 예측 (에이전트 M3-W)

목적: 다음 1주일 계획(시간별 가동 여부 + 생산량)을 넣으면 그 주의 시간별 전력을 예측한다.
      14_plan_advisor.py가 이 모델로 계획 수정안을 평가한다.

10_plan_regression.py(하루 전)와의 차이
    - 예측 시점: 계획 주 시작 직전. 전력·기온 feature는 T-168시간 이전 값만 쓴다 (plan_tools.week_features).
    - 계획 정보: 가동 여부에 더해 시간별 생산량을 넣는다.
    - 학습은 '실제로 실행된 계획'(가동 = 전력 > 41, 생산량 = 기록)으로 한다.
      복제일은 전력이 원본 날짜 것이므로 생산량도 원본 날짜 값으로 맞춘다.
    - 생산량·가동 feature에는 단조 제약(+)을 건다: 다른 조건이 같으면 생산량을 줄였을 때 예측 전력이
      늘어나지 않게 한다. 수정안 평가에서 모델이 "생산량을 줄였더니 전력이 늘었다" 같은 답을 내지 않게 하려는 것이다.

실험 (LightGBM, 7·8월 CV MAE로 선택, test 9/1~9/14는 1회 평가)
    W0 기본          : 1주 전 전력·달력만
    W1 가동계획      : W0 + 가동 여부 계획
    W2 가동+생산계획 : W1 + 생산량 계획 (단조 제약)  <- 본 모델 후보
    W2f              : W2와 같고 단조 제약 없음 (제약의 비용 확인용)
    평가는 두 가지 계획으로 한다.
      - 실행 계획: 그 주에 실제로 실행된 가동·생산량 = "계획이 그대로 지켜질 때"
      - 평소 계획: 사용자 입력 없이 요일별 평소 일정 + 최근 4주 평균 생산량 (14의 템플릿과 같음)

실행:  python preprocessing.py -> python 08_operating_calendar.py -> python 13_week_ahead.py
산출:  results/3_comparison/week_ahead_comparison.csv
       results/3_comparison/week_ahead_sensitivity.csv   (생산량을 바꿨을 때 예측 반응)
       results/2_test_predictions/week_ahead_predictions.csv
       results/4_model_details/week_ahead_oof.csv, week_ahead_<실험>_cv.csv
       results/5_models/week_ahead.joblib + week_ahead_meta.json  (9/14까지 전체로 재학습한 최종 모델)
       results/5_models/week_ahead_until_0831.joblib  (9/1 이전으로만 학습, 9월 백테스트용)
"""
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_absolute_error

import plan_tools as pt
import preprocessing as pp

warnings.filterwarnings("ignore")
SEED = pp.SEED
MAX_ROUNDS, EARLY_STOP = 3000, 100
FIRST_ROW = pd.Timestamp("2021-01-22")       # lag_504 이후

# =============================================================================
# 데이터
# =============================================================================
df, copied, hour_thr, peak_thr = pt.load_history()
executed = pt.executed_plan(df, copied, hour_thr)
idx = df.index[df.index >= FIRST_ROW]
base = pt.week_features(df["power"], df, idx)
plan_exec = pt.plan_features(executed.loc[idx])
BASE = list(base.columns)
ON_COLS = [c for c in plan_exec.columns if c not in pt.PROD_COLS]
FEATURE_SETS = {"W0_기본": BASE, "W1_가동계획": BASE + ON_COLS,
                "W2_가동생산계획": BASE + ON_COLS + pt.PROD_COLS, "W2f_제약없음": BASE + ON_COLS + pt.PROD_COLS}
MONO_UP = {"plan_on", "plan_prod", "plan_prod_roll3", "plan_prod_day"}
data = pd.concat([base, plan_exec], axis=1).assign(
    y=df.loc[idx, "power"], copied=idx.normalize().isin(list(copied)).astype(int))


def leakage_check():
    """원점(origin) 이후 전력·기온을 바꿔도 원점부터 168시간 안의 1주 전 feature는 그대로여야 한다."""
    origin = pp.CV_FOLDS[0][0]
    changed = df.copy()
    num = changed.select_dtypes("number").columns
    changed.loc[changed.index >= origin, num] = changed.loc[changed.index >= origin, num] * 3 + 7
    win = idx[(idx >= origin) & (idx < origin + pd.Timedelta(hours=pt.HORIZON))]
    pd.testing.assert_frame_equal(pt.week_features(changed["power"], changed, win), base.loc[win])
    # 미래 칸을 아예 비워도(NaN) 같아야 한다: 실제 예측 상황
    cut = df.loc[df.index < origin]
    pd.testing.assert_frame_equal(pt.week_features(cut["power"], cut, win), base.loc[win])


leakage_check()


def usual_plan_features(frame_index: pd.DatetimeIndex, cut: pd.Timestamp) -> pd.DataFrame:
    """cut 이전 자료만으로 만든 평소 계획의 feature (주 단위로 만든 것과 같음: 요일·시각 패턴이라 주 경계 무관)."""
    up = pt.usual_plan(df, executed, copied, cut, frame_index)
    return pt.plan_features(up)


CUTS = [s for s, _ in pp.CV_FOLDS] + [pp.TEST_START]
folds = []
for start, end in pp.CV_FOLDS:
    tr = data[data.index < start]
    va = data[(data.index >= start) & (data.index < end) & (data["copied"] == 0)]
    folds.append((start.strftime("%Y-%m"), tr, va, start))
train_all = data[data.index < pp.TEST_START]
test = data[data.index >= pp.TEST_START]
oof_index = pd.concat([va for _, _, va, _ in folds]).index


# =============================================================================
# 모델
# =============================================================================
GRID = [{"num_leaves": nl, "learning_rate": lr, "min_child_samples": mcs, "objective": obj}
        for nl in [15, 31] for lr in [0.03, 0.1] for mcs in [10, 30] for obj in ["regression", "huber"]]


def make(params, feats, n, mono=True):
    cons = [1 if (mono and c in MONO_UP) else 0 for c in feats]
    return LGBMRegressor(**params, n_estimators=n, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                         monotone_constraints=cons, monotone_constraints_method="advanced",
                         random_state=SEED, n_jobs=-1, verbose=-1)


def metrics(y, pred, frame) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    on = frame["plan_on"].to_numpy() == 1
    peak = y >= peak_thr
    g = pd.DataFrame({"y": y, "p": pred}, index=frame.index).groupby(frame.index.normalize())
    dm = g.max()
    hit = (pred >= peak_thr)
    return {"MAE": mean_absolute_error(y, pred), "RMSE": float(np.sqrt(np.mean((y - pred) ** 2))),
            "on_hour_MAE": mean_absolute_error(y[on], pred[on]) if on.any() else np.nan,
            "peak_hour_MAE": mean_absolute_error(y[peak], pred[peak]) if peak.any() else np.nan,
            "daily_max_MAE": float((dm["p"] - dm["y"]).abs().mean()),
            "daily_max_bias": float((dm["p"] - dm["y"]).mean()),
            "peak_recall@179": float((hit & peak).sum() / max(peak.sum(), 1)),
            "peak_precision@179": float((hit & peak).sum() / max(hit.sum(), 1))}


def swap_plan(frame, plan_feats):
    return frame.assign(**plan_feats.loc[frame.index])


def run(name):
    feats, mono = FEATURE_SETS[name], name != "W2f_제약없음"
    rows = []
    for params in GRID:
        preds, its = [], []
        for _, tr, va, _ in folds:
            m = make(params, feats, MAX_ROUNDS, mono)
            m.fit(tr[feats], tr["y"], eval_set=[(va[feats], va["y"])], eval_metric="l1",
                  callbacks=[early_stopping(EARLY_STOP, verbose=False)])
            preds.append(m.predict(va[feats])); its.append(m.best_iteration_)
        oof = np.concatenate(preds)
        rows.append({**params, "cv_MAE": mean_absolute_error(data.loc[oof_index, "y"], oof),
                     "best_iter": int(np.mean(its))})
    cv = pd.DataFrame(rows).sort_values("cv_MAE")
    cv.to_csv(pp.out("detail", f"week_ahead_{name}_cv.csv"), index=False, encoding="utf-8-sig")
    best = cv.iloc[0]
    params = {k: (best[k].item() if hasattr(best[k], "item") else best[k]) for k in GRID[0]}
    n_iter = int(best["best_iter"])

    # 고른 설정으로 fold마다 다시 학습해 OOF를 실행 계획 / 평소 계획 두 가지로 예측 (early stopping 없이 같은 반복 수)
    oof_exec, oof_usual = [], []
    for _, tr, va, start in folds:
        m = make(params, feats, n_iter, mono).fit(tr[feats], tr["y"])
        oof_exec.append(m.predict(va[feats]))
        oof_usual.append(m.predict(swap_plan(va, usual_plan_features(va.index, start))[feats]))
    fold_rows = np.mean([len(tr) for _, tr, _, _ in folds])
    n_final = int(round(n_iter * len(train_all) / fold_rows))
    model = make(params, feats, n_final, mono).fit(train_all[feats], train_all["y"])
    usual_test = swap_plan(test, usual_plan_features(test.index, pp.TEST_START))
    out = {"name": name, "feats": feats, "params": params, "n_iter": n_iter, "n_final": n_final, "mono": mono,
           "model": model, "cv_MAE": float(best["cv_MAE"]),
           "oof_exec": np.concatenate(oof_exec), "oof_usual": np.concatenate(oof_usual),
           "test_exec": model.predict(test[feats]), "test_usual": model.predict(usual_test[feats])}
    print(f"[{name}] {params}, iter {n_iter} -> 최종 {n_final} | CV MAE {out['cv_MAE']:.2f} | "
          f"test MAE 실행계획 {mean_absolute_error(test['y'], out['test_exec']):.2f} / "
          f"평소계획 {mean_absolute_error(test['y'], out['test_usual']):.2f}")
    return out


print(f"[정보] 1주 전 feature {len(BASE)}개 + 가동 계획 {len(ON_COLS)}개 + 생산 계획 {len(pt.PROD_COLS)}개")
print(f"[정보] 학습 {len(train_all)}행 ({train_all.index.min():%m/%d}~), test {len(test)}행, 피크 기준 {peak_thr:g}")
results = {n: run(n) for n in FEATURE_SETS}

# =============================================================================
# 비교표
# =============================================================================
oof_frame = data.loc[oof_index]
rows = []
for r in results.values():
    for plan_kind in ["exec", "usual"]:
        label = "실행 계획" if plan_kind == "exec" else "평소 계획"
        rows.append({"Model": f"{r['name']} / {label}", "split": "test",
                     **metrics(test["y"], r[f"test_{plan_kind}"], test)})
        rows.append({"Model": f"{r['name']} / {label}", "split": "cv(7·8월)",
                     **metrics(oof_frame["y"], r[f"oof_{plan_kind}"], oof_frame)})
for split, frame in [("test", test), ("cv(7·8월)", oof_frame)]:
    rows.append({"Model": "1주 전 같은 시각 (lag_168)", "split": split, **metrics(frame["y"], frame["lag_168"], frame)})
    rows.append({"Model": "3주 같은 시각 평균", "split": split, **metrics(frame["y"], frame["same_hour_3w_mean"], frame)})
table = pd.DataFrame(rows).set_index(["Model", "split"])
table.to_csv(pp.out("cmp", "week_ahead_comparison.csv"), encoding="utf-8-sig")

# =============================================================================
# 본 모델: W2(생산계획, 단조 제약). 생산량 반응 확인
# =============================================================================
main = results["W2_가동생산계획"]


def sensitivity(r):
    """test 가동 시간의 생산량을 0.5배 / 1.5배로 바꿨을 때 예측 전력 변화 (평균, kW)."""
    on = test["plan_on"] == 1
    plan_test = executed.loc[test.index]
    out = {"Model": r["name"]}
    base_pred = r["model"].predict(test[r["feats"]])
    for f in [0.0, 0.5, 1.5, 2.0]:
        changed = pt.plan_features(plan_test.assign(production=plan_test["production"] * f))
        pred = r["model"].predict(swap_plan(test, changed)[r["feats"]])
        out[f"x{f}_mean_change_on_hours"] = float((pred - base_pred)[on].mean())
        out[f"x{f}_mean_change_day_8_17"] = float((pred - base_pred)[on & test.index.hour.isin(range(8, 18))].mean())
    return out


sens = pd.DataFrame([sensitivity(results[n]) for n in ["W2_가동생산계획", "W2f_제약없음"]]).set_index("Model")
sens.to_csv(pp.out("cmp", "week_ahead_sensitivity.csv"), encoding="utf-8-sig")

# 위험 시간 판정용 안전 마진: '예측 + 마진 >= 179'를 피크 경보로 볼 때 CV(7·8월, 실행 계획) F1이 가장 높은 값.
# 모델이 피크를 낮게 예측하는 경향(CV 일 최대 편향 약 -10 kW)을 보정한다. test는 보지 않는다.
y_cv = oof_frame["y"].to_numpy() >= peak_thr
margin_rows = []
for m in np.arange(0, 15.5, 0.5):
    hit = main["oof_exec"] + m >= peak_thr
    tp = (hit & y_cv).sum()
    margin_rows.append({"margin": m, "cv_F1": 2 * tp / max(hit.sum() + y_cv.sum(), 1),
                        "cv_recall": tp / max(y_cv.sum(), 1), "cv_precision": tp / max(hit.sum(), 1)})
margin_table = pd.DataFrame(margin_rows)
margin_table.to_csv(pp.out("detail", "week_ahead_margin_cv.csv"), index=False, encoding="utf-8-sig")
margin = float(margin_table.sort_values(["cv_F1", "margin"], ascending=[False, True]).iloc[0]["margin"])

pd.DataFrame({"Date": test.index, "actual_power": test["y"].to_numpy(),
              "pred_exec_plan": main["test_exec"], "pred_usual_plan": main["test_usual"],
              "pred_no_plan": results["W0_기본"]["test_exec"], "lag_168": test["lag_168"].to_numpy(),
              "plan_on": test["plan_on"].to_numpy(), "plan_prod": test["plan_prod"].to_numpy()}).to_csv(
    pp.out("pred", "week_ahead_predictions.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame({"Date": oof_index, "actual_power": oof_frame["y"].to_numpy(),
              "pred_exec_plan": main["oof_exec"], "pred_usual_plan": main["oof_usual"]}).to_csv(
    pp.out("detail", "week_ahead_oof.csv"), index=False, encoding="utf-8-sig")

# 최종 모델: 9/14까지 전체로 재학습 (앞으로의 주를 예측할 때 쓴다)
n_all = int(round(main["n_iter"] * len(data) / np.mean([len(tr) for _, tr, _, _ in folds])))
final = make(main["params"], main["feats"], n_all, True).fit(data[main["feats"]], data["y"])
joblib.dump(final, pp.out("model", "week_ahead.joblib"))
# 백테스트용: 9/1 이전 자료로만 학습한 같은 모델 (9/1~9/14 계획으로 14를 검증할 때 미래 정보가 섞이지 않게)
joblib.dump(main["model"], pp.out("model", "week_ahead_until_0831.joblib"))
meta = {"script": "13_week_ahead.py", "task": "1주 전 시간별 전력 회귀 (가동·생산 계획 연계)",
        "horizon_hours": pt.HORIZON, "features": main["feats"], "params": main["params"], "n_estimators": n_all,
        "monotone_up": sorted(MONO_UP), "trained_until": str(data.index.max()),
        "backtest_model": {"file": "week_ahead_until_0831.joblib", "trained_until": str(train_all.index.max()),
                           "n_estimators": main["n_final"]},
        "peak_threshold": peak_thr, "hour_on_threshold": hour_thr, "safety_margin_kw": round(margin, 2),
        "margin_basis": "예측 + 마진 >= 피크 기준을 경보로 볼 때 CV(7·8월, 실행 계획) F1 최대 (4_model_details/week_ahead_margin_cv.csv)",
        "cv_MAE": round(main["cv_MAE"], 3),
        "test": {k: round(v, 3) for k, v in metrics(test["y"], main["test_exec"], test).items()},
        "test_usual_plan": {k: round(v, 3) for k, v in metrics(test["y"], main["test_usual"], test).items()}}
with open(pp.out("model", "week_ahead_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

pd.set_option("display.width", 200)
print("\n=== 1주 전 예측 성능 ===")
print(table[["MAE", "RMSE", "on_hour_MAE", "peak_hour_MAE", "daily_max_MAE", "daily_max_bias",
             "peak_recall@179", "peak_precision@179"]].round(2).to_string())
print("\n=== 생산량을 바꿨을 때 test 예측 변화 (kW) ===")
print(sens.round(2).to_string())
print(f"\n[본 모델] W2 / LightGBM, 단조 제약, 안전 마진 {margin:.1f} kW, 최종 학습 ~{data.index.max():%m/%d %H}시")
