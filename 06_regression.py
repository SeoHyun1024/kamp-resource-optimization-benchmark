"""
06 회귀 baseline: 다음 시간 최대수요전력(power_T) 값 자체 예측 (초기 06_regression_baselines.ipynb 재작성)

기존 대비 변경
    - 튜닝: 8월 validation MAE(하계 휴무 포함, val MAE 14.7로 test 6.1과 괴리) -> 7·8월 CV MAE
    - n_estimators: early stopping으로 결정, 손실함수(squared/absolute)도 후보에 포함
    - baseline: Persistence(lag_1) + 1주 전 같은 시각(lag_168) 함께 기록
    - 회귀 예측으로 피크 판정: "예측값 >= cutoff" 의 cutoff를 CV OOF F1로 선택해
      분류 모델들과 같은 표에 기록
"""
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

import common as cm
import preprocessing as pp

warnings.filterwarnings("ignore", category=UserWarning)

ds, FEATURES, peak_thr = pp.load_dataset(log=lambda *_: None)
train_df, test_df = cm.final_split(ds)
folds = cm.cv_folds(ds)
fold_rows = np.mean([len(tr) for _, tr, _ in folds])
CUTOFF_GRID = np.arange(165, 190, 1.0)
MAX_ROUNDS, EARLY_STOP = 3000, 100


def report(y, pred) -> dict:
    return {"MAE": mean_absolute_error(y, pred), "RMSE": float(np.sqrt(mean_squared_error(y, pred)))}


def peak_hour_mae(frame, pred):
    mask = frame["label"].to_numpy() == 1
    return mean_absolute_error(frame["target_power"].to_numpy()[mask], np.asarray(pred)[mask])


rows = []
for name, col in [("Persistence (lag_1)", "lag_1"), ("1주 전 같은 시각 (lag_168)", "lag_168")]:
    rows.append({"Model": name, **report(test_df["target_power"], test_df[col]),
                 "peak_hour_MAE": peak_hour_mae(test_df, test_df[col])})

MODELS = {
    "xgboost_reg": {
        "grid": [{"max_depth": d, "learning_rate": lr, "min_child_weight": mcw, "objective": obj}
                 for d in [3, 5, 7] for lr in [0.03, 0.1] for mcw in [1, 5]
                 for obj in ["reg:squarederror", "reg:absoluteerror"]],
        "make": lambda p, n, es=None: XGBRegressor(**p, n_estimators=n, subsample=0.8, colsample_bytree=0.8,
                                                   tree_method="hist", random_state=cm.SEED, n_jobs=-1,
                                                   early_stopping_rounds=es),
    },
    "lightgbm_reg": {
        "grid": [{"num_leaves": nl, "learning_rate": lr, "min_child_samples": mcs, "objective": obj}
                 for nl in [7, 15, 31] for lr in [0.03, 0.1] for mcs in [10, 30]
                 for obj in ["regression", "regression_l1"]],
        "make": lambda p, n, es=None: LGBMRegressor(**p, n_estimators=n, subsample=0.8, subsample_freq=1,
                                                    colsample_bytree=0.8, random_state=cm.SEED, n_jobs=-1, verbose=-1),
    },
}


def fit_fold(key, params, tr, va):
    spec = MODELS[key]
    if key == "xgboost_reg":
        m = spec["make"](params, MAX_ROUNDS, EARLY_STOP)
        m.fit(tr[FEATURES], tr["target_power"], eval_set=[(va[FEATURES], va["target_power"])], verbose=False)
        return m.predict(va[FEATURES]), m.best_iteration + 1
    m = spec["make"](params, MAX_ROUNDS)
    m.fit(tr[FEATURES], tr["target_power"], eval_set=[(va[FEATURES], va["target_power"])],
          eval_metric="l1", callbacks=[early_stopping(EARLY_STOP, verbose=False)])
    return m.predict(va[FEATURES]), m.best_iteration_


for key, spec in MODELS.items():
    cv_rows = []
    for params in spec["grid"]:
        ys, ps, its = [], [], []
        for _, tr, va in folds:
            pred, it = fit_fold(key, params, tr, va)
            ys.append(va); ps.append(pred); its.append(it)
        oof = pd.concat(ys)
        pred = np.concatenate(ps)
        cv_rows.append({**params, "cv_MAE": mean_absolute_error(oof["target_power"], pred),
                        "best_iter": int(np.mean(its)), "_oof": (oof, pred)})
    cv = pd.DataFrame(cv_rows).sort_values("cv_MAE")
    cv.drop(columns="_oof").to_csv(cm.out("detail", f"{key}_cv.csv"), index=False)
    best = cv.iloc[0]
    params = {k: best[k] for k in spec["grid"][0]}
    params = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
              for k, v in params.items()}
    n_final = int(round(best["best_iter"] * len(train_df) / fold_rows))

    # 회귀값 -> 피크 판정 cutoff (CV OOF F1 최대)
    oof, oof_pred = best["_oof"]
    f1s = [cm.evaluate(oof["label"], (oof_pred >= c).astype(float), 0.5)["F1"] for c in CUTOFF_GRID]
    cutoff = float(CUTOFF_GRID[int(np.argmax(f1s))])

    model = spec["make"](params, n_final)
    model.fit(train_df[FEATURES], train_df["target_power"])
    pred_test = model.predict(test_df[FEATURES])
    rows.append({"Model": key, **report(test_df["target_power"], pred_test),
                 "peak_hour_MAE": peak_hour_mae(test_df, pred_test), "cv_MAE": best["cv_MAE"],
                 "params": f"{params}, n_estimators={n_final}"})
    m = cm.save_results(key, test_df, pred_test, cutoff, score_col="predicted_power",
                        extra={"method": f"예측값 >= {cutoff:.0f} (CV 선택)", "cv_F1": max(f1s)})
    print(f"[{key}] {params}, n={n_final} | CV MAE {best['cv_MAE']:.3f} | "
          f"test MAE {rows[-1]['MAE']:.3f}, RMSE {rows[-1]['RMSE']:.3f}, 피크시간 MAE {rows[-1]['peak_hour_MAE']:.2f}")
    print(f"    피크 판정(예측 >= {cutoff:.0f}): {cm.fmt(m)}")
    cm.save_model(key, model, {
        "script": "06_regression.py", "task": "다음 시간 power 회귀 + cutoff로 피크 판정",
        "target": "target_power (연속값)", "decision_rule": f"predict(X) >= {cutoff}", "peak_cutoff": cutoff,
        "calendar_encoding": "int", "features": FEATURES,
        "params": {**params, "n_estimators": n_final},
        "cv": {"MAE": float(best["cv_MAE"]), "F1": float(max(f1s))},
        "test": {**cm.test_summary(m), "MAE": float(rows[-1]["MAE"]), "RMSE": float(rows[-1]["RMSE"])},
    })

table = pd.DataFrame(rows).set_index("Model")
base = table.loc["Persistence (lag_1)", "MAE"]
table["MAE 개선율(vs lag_1)"] = (base - table["MAE"]) / base
table.to_csv(cm.out("cmp", "regression_comparison.csv"))
print("\n=== Test 회귀 성능 ===")
print(table[["MAE", "RMSE", "peak_hour_MAE", "MAE 개선율(vs lag_1)"]].round(4).to_string())
print("기존 노트북: XGBoost MAE 6.062 / LightGBM MAE 6.654 / Persistence 14.521")
