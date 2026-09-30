"""
03 XGBoost 피크 위험 분류 (초기 03_xgboost.ipynb 재작성)

기존 대비 변경
    - 전처리/feature: preprocessing.py
    - scale_pos_weight: 학습 구간마다 neg/pos 를 다시 계산 (fold마다 양성 비율이 다름)
    - n_estimators: 고정 grid 대신 early stopping(CV fold validation PR-AUC)으로 결정,
      최종 학습 시에는 두 fold best_iteration 평균 x (최종 학습량/fold 학습량) 보정
    - 튜닝·threshold: 7·8월 확장창 CV OOF
    - 최종 학습: 9/1 이전 전체
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import common as cm
import preprocessing as pp

ds, FEATURES, peak_thr = pp.load_dataset(log=lambda *_: None)
train_df, test_df = cm.final_split(ds)
print(f"[정보] feature {len(FEATURES)}개, train {len(train_df)} / test {len(test_df)} (피크 {test_df['label'].sum()}건)")

BASE = dict(objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
            subsample=0.8, colsample_bytree=0.8, random_state=cm.SEED, n_jobs=-1)
MAX_ROUNDS, EARLY_STOP = 2000, 100


def spw(frame: pd.DataFrame) -> float:
    return float((frame["label"] == 0).sum() / frame["label"].sum())


grid = [
    {"max_depth": d, "learning_rate": lr, "min_child_weight": mcw, "use_spw": s}
    for d in [3, 4, 6]
    for lr in [0.03, 0.1]
    for mcw in [1, 5]
    for s in [False, True]
]


def make_model(params, frame, n_estimators, early_stop=None):
    p = dict(params)
    use_spw = p.pop("use_spw")
    return XGBClassifier(**BASE, **p, n_estimators=n_estimators,
                         scale_pos_weight=spw(frame) if use_spw else 1.0,
                         early_stopping_rounds=early_stop)


def fit_predict(params, tr, va):
    model = make_model(params, tr, MAX_ROUNDS, EARLY_STOP)
    model.fit(tr[FEATURES], tr["label"], eval_set=[(va[FEATURES], va["label"])], verbose=False)
    return model.predict_proba(va[FEATURES])[:, 1], model.best_iteration + 1


print(f"\n[정보] XGBoost CV 튜닝 ({len(grid)}개 조합 x 2 fold, early stopping)")
table, best, (oof_y, oof_s), best_params = cm.run_cv(ds, grid, fit_predict)
table.to_csv(cm.out("detail", "xgboost_cv.csv"), index=False)
threshold = best["cv_threshold"]
cm.save_oof("xgboost", ds, oof_y, oof_s)

# fold 평균 학습 행수 대비 최종 학습 행수만큼 트리 수를 비례 보정
fold_rows = np.mean([len(tr) for _, tr, _ in cm.cv_folds(ds)])
n_final = int(round(best["best_iter"] * len(train_df) / fold_rows))
print(f"[선택] {best_params} | CV PR-AUC {best['cv_PR-AUC']:.4f} (fold {best['fold_PR-AUC']}) | "
      f"CV F1 {best['cv_F1']:.4f} @ {threshold} | best_iter {best['best_iter']} -> 최종 {n_final}")

xgb = make_model(best_params, train_df, n_final)
xgb.fit(train_df[FEATURES], train_df["label"], verbose=False)
test_proba = xgb.predict_proba(test_df[FEATURES])[:, 1]
metrics = cm.save_results("xgboost", test_df, test_proba, threshold,
                          extra={"method": f"{best_params}, n_estimators={n_final}",
                                 "cv_PR-AUC": best["cv_PR-AUC"], "cv_F1": best["cv_F1"]})
print(f"\n=== Test (2021-09-01~09-14) ===\n[xgboost] {cm.fmt(metrics)}")
xgb.save_model(cm.out("model", "xgboost_classifier.json"))

gain = xgb.get_booster().get_score(importance_type="gain")
imp = pd.Series({f: gain.get(f, 0.0) for f in FEATURES}).sort_values(ascending=False)
imp = imp / imp.sum()
imp.to_csv(cm.out("detail", "xgboost_importance.csv"), header=["gain_ratio"])
print("\n[gain importance top 10]\n" + imp.head(10).round(4).to_string())
