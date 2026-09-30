"""
04 CatBoost 피크 위험 분류 (초기 04_catboost.ipynb 재작성)

기존 대비 변경
    - 범주형: hour, day_of_week 만 native categorical (month/Weekend/Vacation 제거
      -> 기존 노트북 12-2절에서 지적된 "9월 month 범주 offset" 문제 자체가 사라짐)
    - early stopping·grid·threshold를 8월 하나에 몰아 쓰던 것을 7·8월 CV로 분산
    - 최종 학습: 9/1 이전 전체, iterations = CV best_iteration 평균 x 학습량 보정
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

import common as cm
import preprocessing as pp

ds, FEATURES, peak_thr = pp.load_dataset(log=lambda *_: None)
train_df, test_df = cm.final_split(ds)
CAT = pp.CATEGORICAL
print(f"[정보] feature {len(FEATURES)}개 (categorical {CAT}), train {len(train_df)} / test {len(test_df)}")

MAX_ITER, EARLY_STOP = 2000, 100


def pool(frame, with_label=True):
    X = frame[FEATURES].astype({c: "int64" for c in CAT}).astype({c: "string" for c in CAT})
    return Pool(X, frame["label"] if with_label else None, cat_features=CAT)


def make_model(params, iterations):
    p = dict(params)
    acw = p.pop("auto_class_weights")
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC" if acw is None else "PRAUC:use_weights=false",
        iterations=iterations, random_seed=cm.SEED, allow_writing_files=False,
        verbose=False, thread_count=-1, auto_class_weights=acw, **p,
    )


grid = [
    {"depth": d, "learning_rate": lr, "l2_leaf_reg": l2, "auto_class_weights": acw}
    for d in [4, 6]
    for lr in [0.03, 0.1]
    for l2 in [3, 10]
    for acw in [None, "Balanced"]
]


def fit_predict(params, tr, va):
    model = make_model(params, MAX_ITER)
    model.fit(pool(tr), eval_set=pool(va), early_stopping_rounds=EARLY_STOP, use_best_model=True)
    return model.predict_proba(pool(va, False))[:, 1], model.get_best_iteration() + 1


print(f"\n[정보] CatBoost CV 튜닝 ({len(grid)}개 조합 x 2 fold, early stopping)")
table, best, (oof_y, oof_s), best_params = cm.run_cv(ds, grid, fit_predict)
table.to_csv(cm.out("detail", "catboost_cv.csv"), index=False)
threshold = best["cv_threshold"]
cm.save_oof("catboost", ds, oof_y, oof_s)
fold_rows = np.mean([len(tr) for _, tr, _ in cm.cv_folds(ds)])
n_final = int(round(best["best_iter"] * len(train_df) / fold_rows))
print(f"[선택] {best_params} | CV PR-AUC {best['cv_PR-AUC']:.4f} (fold {best['fold_PR-AUC']}) | "
      f"CV F1 {best['cv_F1']:.4f} @ {threshold} | best_iter {best['best_iter']} -> 최종 {n_final}")

cb = make_model(best_params, n_final)
cb.fit(pool(train_df))
test_proba = cb.predict_proba(pool(test_df, False))[:, 1]
metrics = cm.save_results("catboost", test_df, test_proba, threshold,
                          extra={"method": f"{best_params}, iterations={n_final}",
                                 "cv_PR-AUC": best["cv_PR-AUC"], "cv_F1": best["cv_F1"]})
print(f"\n=== Test (2021-09-01~09-14) ===\n[catboost] {cm.fmt(metrics)}")
cm.save_model("catboost_classifier", cb, {
    "script": "04_catboost.py", "task": "피크 위험 확률 분류",
    "decision_rule": f"predict_proba[:, 1] >= {threshold}", "classification_threshold": threshold,
    "calendar_encoding": "categorical (hour, day_of_week -> string, cat_features)",
    "cat_features": CAT, "features": FEATURES,
    "params": {**best_params, "iterations": n_final},
    "cv": {"PR-AUC": best["cv_PR-AUC"], "F1": best["cv_F1"]}, "test": cm.test_summary(metrics),
}, kind="catboost")

imp = pd.Series(cb.get_feature_importance(pool(train_df), type="ShapValues")[:, :-1].__abs__().mean(axis=0),
                index=FEATURES).sort_values(ascending=False)
imp.to_csv(cm.out("detail", "catboost_shap_importance.csv"), header=["mean_abs_shap"])
print("\n[SHAP(평균 |기여도|) top 10, train]\n" + imp.head(10).round(4).to_string())
