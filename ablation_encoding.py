"""
보조 실험: 달력 변수(hour, day_of_week) 인코딩 방식별 성능 비교 (이슈 #2 체크리스트)

비교 방식
    int    : 정수 그대로 (hour 0~23, day_of_week 0~6)          <- 현재 기본값
    onehot : one-hot (hour 24개 + 요일 7개 = 31개 컬럼, 원래 2개 대체)
    cyclic : sin/cos (hour_sin, hour_cos, dow_sin, dow_cos 4개 컬럼)
    int+cyclic : 정수 + sin/cos 둘 다

대상 모델: Random Forest, XGBoost
    (CatBoost는 범주형 native 처리, RNN은 이미 sin/cos 입력이라 제외)
    하이퍼파라미터는 02·03 스크립트가 CV로 고른 값으로 고정하고 인코딩만 바꾼다.

평가: 7·8월 확장창 CV의 OOF PR-AUC / F1(OOF 최적 threshold), 모델 seed 3개 평균
      + 9/1 이전 전체로 재학습한 test F1 / PR-AUC (threshold는 OOF에서 고른 값, seed 3개 평균)
산출: results/4_model_details/encoding_ablation.csv (seed별 상세)
      results/3_comparison/encoding_ablation_summary.csv
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

import common as cm
import preprocessing as pp

warnings.filterwarnings("ignore")
SEEDS = [42, 7, 123]
ENCODINGS = ["int", "onehot", "cyclic", "int+cyclic"]

# 02·03 스크립트가 CV로 선택한 설정 (results/3_comparison/model_comparison.csv 의 method 열)
RF_PARAMS = {"max_depth": None, "min_samples_leaf": 5, "max_features": "sqrt", "class_weight": "balanced_subsample"}
XGB_PARAMS = {"max_depth": 3, "learning_rate": 0.03, "min_child_weight": 5, "n_estimators": 328}

ds, BASE_FEATURES, peak_thr = pp.load_dataset(log=lambda *_: None)
CAL = ["hour", "day_of_week"]
NON_CAL = [f for f in BASE_FEATURES if f not in CAL]


def encode(frame: pd.DataFrame, how: str) -> tuple[pd.DataFrame, list[str]]:
    """달력 컬럼만 바꾼 feature 표를 만든다 (행·라벨·분할은 그대로)."""
    parts = [frame[NON_CAL]]
    if how in ("int", "int+cyclic"):
        parts.append(frame[CAL])
    if how == "onehot":
        # 전체 범주를 고정해 fold마다 컬럼이 달라지지 않게 한다 (hour 0~23, 요일 0~6)
        h = pd.get_dummies(pd.Categorical(frame["hour"], categories=range(24)), prefix="hour").astype(int)
        d = pd.get_dummies(pd.Categorical(frame["day_of_week"], categories=range(7)), prefix="dow").astype(int)
        parts += [h.set_index(frame.index), d.set_index(frame.index)]
    if how in ("cyclic", "int+cyclic"):
        parts.append(pd.DataFrame({
            "hour_sin": np.sin(2 * np.pi * frame["hour"] / 24), "hour_cos": np.cos(2 * np.pi * frame["hour"] / 24),
            "dow_sin": np.sin(2 * np.pi * frame["day_of_week"] / 7), "dow_cos": np.cos(2 * np.pi * frame["day_of_week"] / 7),
        }, index=frame.index))
    X = pd.concat(parts, axis=1)
    return X, list(X.columns)


def fit_predict(model_name, seed, Xtr, ytr, Xte):
    if model_name == "random_forest":
        m = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1, **RF_PARAMS)
    else:
        m = XGBClassifier(**XGB_PARAMS, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                          random_state=seed, n_jobs=-1)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


rows = []
for how in ENCODINGS:
    X_all, feats = encode(ds, how)
    for model in ["random_forest", "xgboost"]:
        for seed in SEEDS:
            ys, ss = [], []
            for _, tr, va in cm.cv_folds(ds):
                ys.append(va["label"].to_numpy())
                ss.append(fit_predict(model, seed, X_all.loc[tr.index], tr["label"], X_all.loc[va.index]))
            y, s = np.concatenate(ys), np.concatenate(ss)
            thr = cm.best_threshold(y, s)
            train_df, test_df = cm.final_split(ds)
            test_s = fit_predict(model, seed, X_all.loc[train_df.index], train_df["label"], X_all.loc[test_df.index])
            t = cm.evaluate(test_df["label"], test_s, thr)
            rows.append({"encoding": how, "model": model, "seed": seed, "n_features": len(feats),
                         "cv_PR-AUC": average_precision_score(y, s), "cv_F1": cm.evaluate(y, s, thr)["F1"],
                         "threshold": thr, "test_F1": t["F1"], "test_PR-AUC": t["PR-AUC"],
                         "test_Precision": t["Precision"], "test_Recall": t["Recall"]})
        part = pd.DataFrame([r for r in rows if r["encoding"] == how and r["model"] == model])
        print(f"{how:<11} {model:<14} feature {len(feats):>2}개 | CV PR-AUC {part['cv_PR-AUC'].mean():.4f} ± {part['cv_PR-AUC'].std():.4f}"
              f" | CV F1 {part['cv_F1'].mean():.4f} | test F1 {part['test_F1'].mean():.4f} | test PR-AUC {part['test_PR-AUC'].mean():.4f}")

detail = pd.DataFrame(rows)
detail.to_csv(cm.out("detail", "encoding_ablation.csv"), index=False)
metrics = ["n_features", "cv_PR-AUC", "cv_F1", "test_F1", "test_PR-AUC", "test_Precision", "test_Recall"]
summary = detail.groupby(["model", "encoding"], sort=False)[metrics].mean()
summary["cv_PR-AUC_std"] = detail.groupby(["model", "encoding"], sort=False)["cv_PR-AUC"].std()
summary.to_csv(cm.out("cmp", "encoding_ablation_summary.csv"))

print("\n=== 요약 (seed 3개 평균) ===")
print(summary.round(4).to_string())
for model in ["random_forest", "xgboost"]:
    s = summary.loc[model, "cv_PR-AUC"]
    print(f"[{model}] CV PR-AUC 최고 {s.idxmax()} {s.max():.4f} / 최저 {s.idxmin()} {s.min():.4f} / 범위 {s.max() - s.min():.4f}")
