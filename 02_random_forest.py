"""
02 Random Forest 피크 위험 분류 (초기 03_xgboost.ipynb 안의 RF baseline 재작성)

X_T-1 (전력 lag·15분 값·전일 가동 정보·외생변수·시간/요일) -> P(power_T >= 179)

기존 대비 변경
    - 전처리/feature: preprocessing.py (38개 feature, month/Weekend/Vacation 제거)
    - 튜닝: 8월 1개월 validation -> 7·8월 확장창 CV의 OOF PR-AUC
    - threshold: CV OOF F1 최대값
    - 최종 학습: 9/1 이전 전체(기존은 7/31까지)
    - baseline 2종(Persistence lag_1, 1주 전 같은 시각 lag_168)을 같은 test로 함께 기록
    - 달력 인코딩: hour·day_of_week 정수 + sin/cos (ablation_encoding.py로 선택, feature 42개)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import common as cm
import preprocessing as pp

N_TREES = 500

# 단계1: 데이터 ------------------------------------------------------------------
ds, FEATURES, peak_thr = pp.load_dataset(log=lambda *_: None)
# 달력 인코딩: 정수 + sin/cos (ablation_encoding.py에서 CV로 선택, 보고서 5.4절)
CALENDAR_ENCODING = "int+cyclic"
if CALENDAR_ENCODING == "int+cyclic":
    ds, FEATURES = pp.add_cyclic_calendar(ds, FEATURES)
train_df, test_df = cm.final_split(ds)
print(f"[정보] feature {len(FEATURES)}개, 피크 임계값 {peak_thr:.0f}, train {len(train_df)} / test {len(test_df)} (피크 {test_df['label'].sum()}건)")

# 단계2: 규칙 기반 baseline -------------------------------------------------------
for name, col in [("persistence", "lag_1"), ("persistence_lag168", "lag_168")]:
    m = cm.save_results(name, test_df, test_df[col].to_numpy(), peak_thr, extra={"method": f"{col} >= {peak_thr:.0f}"}, score_col="score")
    print(f"[baseline] {name:<19} {cm.fmt(m)}")

# 단계3: CV 튜닝 -----------------------------------------------------------------
grid = [
    {"max_depth": d, "min_samples_leaf": leaf, "max_features": mf, "class_weight": cw}
    for d in [None, 12]
    for leaf in [1, 5, 10]
    for mf in ["sqrt", 0.5]
    for cw in [None, "balanced_subsample"]
]


def fit_predict(params, tr, va):
    model = RandomForestClassifier(n_estimators=300, random_state=cm.SEED, n_jobs=-1, **params)
    model.fit(tr[FEATURES], tr["label"])
    return model.predict_proba(va[FEATURES])[:, 1], None


print(f"\n[정보] RF CV 튜닝 ({len(grid)}개 조합 x 2 fold)")
table, best, (oof_y, oof_s), best_params = cm.run_cv(ds, grid, fit_predict)
table.to_csv(cm.out("detail", "random_forest_cv.csv"), index=False)
threshold = best["cv_threshold"]
cm.save_oof("random_forest", ds, oof_y, oof_s)
print(f"[선택] {best_params} | CV PR-AUC {best['cv_PR-AUC']:.4f} (fold {best['fold_PR-AUC']}) | CV F1 {best['cv_F1']:.4f} @ threshold {threshold}")

# 단계4: 최종 학습(9/1 이전 전체) + test 1회 평가 -------------------------------------
rf = RandomForestClassifier(n_estimators=N_TREES, random_state=cm.SEED, n_jobs=-1, **best_params)
rf.fit(train_df[FEATURES], train_df["label"])
test_proba = rf.predict_proba(test_df[FEATURES])[:, 1]
metrics = cm.save_results("random_forest", test_df, test_proba, threshold,
                          extra={"method": f"{best_params}, calendar={CALENDAR_ENCODING}", "cv_PR-AUC": best["cv_PR-AUC"], "cv_F1": best["cv_F1"]})
print(f"\n=== Test (2021-09-01~09-14) ===\n[random_forest] {cm.fmt(metrics)}")
cm.save_model("random_forest", rf, {
    "script": "02_random_forest.py", "task": "피크 위험 확률 분류",
    "decision_rule": f"predict_proba[:, 1] >= {threshold}", "classification_threshold": threshold,
    "calendar_encoding": CALENDAR_ENCODING, "features": FEATURES,
    "params": {**best_params, "n_estimators": N_TREES},
    "cv": {"PR-AUC": best["cv_PR-AUC"], "F1": best["cv_F1"]}, "test": cm.test_summary(metrics),
})

imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
imp.to_csv(cm.out("detail", "random_forest_importance.csv"), header=["importance"])
print("\n[변수중요도 top 10]\n" + imp.head(10).round(4).to_string())
