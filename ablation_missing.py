"""
보조 실험: 결측 처리 방식별 모델 성능 비교 (이슈 #2 체크리스트)

비교 방식 (preprocessing.MISSING_STRATEGIES)
    interp : 시간 보간(앞뒤 값)                          <- 현재 기본값
    ffill  : 직전 관측값 유지 (실시간 운영에서 그대로 재현 가능, 미래 값 미사용)
    zero   : 0으로 채움 (기존 01 노트북 fillna(0))
    native : 결측 그대로 -> 모델 내장 결측 처리 (XGBoost·CatBoost·RF 모두 지원)
    ※ 모든 방식에서 '생산량 0 구간의 공장인원 결측'은 0으로 채운다(native 제외).

대상 모델: Random Forest, XGBoost, CatBoost
    (Isolation Forest는 결측 입력을 지원하지 않고, RNN 입력에는 결측 컬럼이 없어 제외)
    하이퍼파라미터는 02~04 스크립트가 CV로 고른 값으로 고정하고 결측 처리만 바꾼다.

실험 A. 실제 데이터 그대로
    원자료 결측은 21셀(풍속 3, 강수량 1, 공장인원 17)뿐이고 test 구간(9/1~9/14)에는 없다.
    -> 방식 간 차이가 거의 없을 것으로 예상되며, 그것을 수치로 확인한다.
실험 B. 결측을 인위적으로 늘린 강건성 시험 (운영 중 센서 누락 상황 가정)
    외생변수 6개(기온, 풍속, 습도, 강수량, 생산량, 공장인원)에 전 구간(test 포함) 결측 주입
      - random : 셀 단위 무작위 10%
      - block  : 6시간 연속 누락 구간을 컬럼마다 전체 시간의 약 10%만큼
    결측 주입 시드 3개 평균.

평가: 7·8월 확장창 CV의 OOF PR-AUC / F1(OOF 최적 threshold)
      + 9/1 이전 전체로 재학습한 test F1 / PR-AUC (threshold는 OOF에서 고른 값)
산출: results/4_model_details/missing_ablation_real.csv, missing_ablation_simulated.csv
      results/3_comparison/missing_ablation_summary.csv
"""
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

import common as cm
import preprocessing as pp

warnings.filterwarnings("ignore")
STRATEGIES = list(pp.MISSING_STRATEGIES)
MASK_COLS = ["기온", "풍속", "습도", "강수량", "생산량", "공장인원"]
MASK_RATE, BLOCK_HOURS = 0.10, 6
MASK_SEEDS = [0, 1, 2]
CAT = pp.CATEGORICAL

# 02~04 스크립트가 CV로 선택한 설정 (results/3_comparison/model_comparison.csv 의 method 열)
RF_PARAMS = {"max_depth": None, "min_samples_leaf": 5, "max_features": "sqrt", "class_weight": "balanced_subsample"}
XGB_PARAMS = {"max_depth": 3, "learning_rate": 0.03, "min_child_weight": 5, "n_estimators": 328}
CB_PARAMS = {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 10, "iterations": 228}


def fit_predict(model_name, tr, te, features):
    if model_name == "random_forest":
        m = RandomForestClassifier(n_estimators=300, random_state=cm.SEED, n_jobs=-1, **RF_PARAMS)
        m.fit(tr[features], tr["label"])
        return m.predict_proba(te[features])[:, 1]
    if model_name == "xgboost":
        m = XGBClassifier(**XGB_PARAMS, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                          random_state=cm.SEED, n_jobs=-1)
        m.fit(tr[features], tr["label"])
        return m.predict_proba(te[features])[:, 1]
    to_pool = lambda f, lab=True: Pool(f[features].astype({c: "int64" for c in CAT}).astype({c: "string" for c in CAT}),
                                       f["label"] if lab else None, cat_features=CAT)
    m = CatBoostClassifier(**CB_PARAMS, loss_function="Logloss", random_seed=cm.SEED,
                           allow_writing_files=False, verbose=False, thread_count=-1)
    m.fit(to_pool(tr))
    return m.predict_proba(to_pool(te, False))[:, 1]


def make_dataset(raw, missing):
    df = pp.clean(raw, log=lambda *_: None, missing=missing)
    ds, features = pp.build_features(df)
    pp.leakage_checks(ds, df, features)
    thr = pp.peak_threshold_legacy(df)
    ds["label"] = (ds["target_power"] >= thr).astype(int)
    return ds, features


def evaluate_all(ds, features, model_name):
    ys, ss = [], []
    for _, tr, va in cm.cv_folds(ds):
        ys.append(va["label"].to_numpy()); ss.append(fit_predict(model_name, tr, va, features))
    y, s = np.concatenate(ys), np.concatenate(ss)
    thr = cm.best_threshold(y, s)
    train_df, test_df = cm.final_split(ds)
    test_s = fit_predict(model_name, train_df, test_df, features)
    t = cm.evaluate(test_df["label"], test_s, thr)
    return {"cv_PR-AUC": average_precision_score(y, s), "cv_F1": cm.evaluate(y, s, thr)["F1"],
            "threshold": thr, "test_F1": t["F1"], "test_PR-AUC": t["PR-AUC"],
            "test_Recall": t["Recall"], "test_Precision": t["Precision"]}


def inject(raw, pattern, seed):
    rng = np.random.default_rng(seed)
    out = raw.copy()
    n = len(out)
    for col in MASK_COLS:
        if pattern == "random":
            mask = rng.random(n) < MASK_RATE
        else:
            mask = np.zeros(n, dtype=bool)
            for start in rng.integers(0, n - BLOCK_HOURS, int(n * MASK_RATE / BLOCK_HOURS)):
                mask[start:start + BLOCK_HOURS] = True
        out.loc[mask, col] = np.nan
    return out


MODELS = ["random_forest", "xgboost", "catboost"]
raw = pp.load_raw()

# --- 실험 A: 실제 결측 ------------------------------------------------------------------
print("=== 실험 A: 실제 데이터 (결측 21셀, test 구간 결측 0) ===")
rows_a = []
for missing in STRATEGIES:
    ds, F = make_dataset(raw, missing)
    for model in MODELS:
        r = evaluate_all(ds, F, model)
        rows_a.append({"missing": missing, "model": model, **r})
        print(f"  {missing:<7} {model:<14} CV PR-AUC {r['cv_PR-AUC']:.4f} F1 {r['cv_F1']:.4f} | test F1 {r['test_F1']:.4f} PR-AUC {r['test_PR-AUC']:.4f}")
real = pd.DataFrame(rows_a)
real.to_csv(cm.out("detail", "missing_ablation_real.csv"), index=False)

# --- 실험 B: 결측 주입 ------------------------------------------------------------------
print(f"\n=== 실험 B: 외생변수 {len(MASK_COLS)}개에 결측 {MASK_RATE:.0%} 주입 (test 포함, 시드 {len(MASK_SEEDS)}개) ===")
rows_b = []
for pattern in ["random", "block"]:
    for seed in MASK_SEEDS:
        masked = inject(raw, pattern, seed)
        for missing in STRATEGIES:
            ds, F = make_dataset(masked, missing)
            for model in MODELS:
                rows_b.append({"pattern": pattern, "seed": seed, "missing": missing, "model": model,
                               **evaluate_all(ds, F, model)})
    part = pd.DataFrame([r for r in rows_b if r["pattern"] == pattern])
    agg = part.groupby(["model", "missing"])[["cv_PR-AUC", "cv_F1", "test_F1", "test_PR-AUC"]].mean()
    print(f"\n[{pattern}]\n" + agg.round(4).to_string())
sim = pd.DataFrame(rows_b)
sim.to_csv(cm.out("detail", "missing_ablation_simulated.csv"), index=False)

# --- 요약 -------------------------------------------------------------------------------
metric = ["cv_PR-AUC", "cv_F1", "test_F1"]
summary = pd.concat({
    "A_real": real.set_index(["model", "missing"])[metric],
    "B_random10%": sim[sim.pattern == "random"].groupby(["model", "missing"])[metric].mean(),
    "B_block10%": sim[sim.pattern == "block"].groupby(["model", "missing"])[metric].mean(),
}, axis=1)
summary.to_csv(cm.out("cmp", "missing_ablation_summary.csv"))
print("\n=== 요약: 모델별 방식 간 CV PR-AUC 범위 (최대 - 최소) ===")
for exp in ["A_real", "B_random10%", "B_block10%"]:
    rng_ = summary[exp]["cv_PR-AUC"].groupby(level="model").agg(lambda s: s.max() - s.min())
    best = summary[exp]["cv_PR-AUC"].groupby(level="model").idxmax().map(lambda t: t[1])
    print(f"  {exp:<12} " + " | ".join(f"{m}: 범위 {rng_[m]:.4f}, 최고 {best[m]}" for m in MODELS))
