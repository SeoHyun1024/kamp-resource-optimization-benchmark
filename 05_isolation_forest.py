"""
05 Isolation Forest 비지도 이상탐지 (초기 05_isolation_forest.ipynb 재작성)

기존 노트북의 구조적 문제
    - 30개 feature 전체(시간·요일·외생변수 포함)로 학습 -> test에서 "이상"으로 잡힌 10건이
      월요일 가동 시작(9/6, 9/13 08~14시)·토요일 가동 종료(9/11 08시) 같은
      "평소와 다른 달력/전환 시점"이었다. 피크(=높은 전력)는 평일 주간에 자주 나오는
      정상 상태라 IF 입장에서는 이상치가 아니다.
    - contamination 4개 값(0.01~0.10)만 탐색 -> 실제 피크 비율(10~14%) 부근을 거의 못 봄.

변경 (여전히 학습에 라벨은 쓰지 않는다. 라벨은 CV에서 설정 "선택"에만 사용 = 기존과 동일 원칙)
    - 실험 A: 기존과 같은 "전체 feature"
    - 실험 B: 전력 동특성 feature만 (lag·15분 값·rolling·전일/주간 같은 시각)
    - 실험 C: B + "가동 시간대(전일 가동 & 08~18시)" 데이터로만 학습 -> 가동 중
             평소보다 높은 전력을 이상으로 보게 함
    - 판정 경계: contamination 대신 train score 분위수 grid(1%~45%)를 CV OOF F1로 선택
    - 점수 방향: -decision_function (클수록 이상). 추가로 C에서는 전력 수준이 낮은
      쪽의 이상(비가동 등)을 배제하기 위해 score에 "lag_1이 train 중앙값보다 높을 때만"
      부호를 유지하는 방향성 보정을 옵션으로 비교한다.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

import common as cm
import preprocessing as pp

ds, FEATURES, peak_thr = pp.load_dataset(log=lambda *_: None)
train_df, test_df = cm.final_split(ds)

POWER_FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_24", "lag_168", "lag_336",
    "q15_lag1", "q30_lag1", "q45_lag1", "q60_lag1", "avg_lag1", "intra_hour_slope", "diff_1",
    "roll_mean_3", "roll_mean_24", "roll_max_24", "roll_std_24",
    "same_hour_2w_mean", "same_hour_2w_max", "last_op_day_same_hour", "max_since_midnight",
]
IF_PARAMS = dict(n_estimators=300, max_samples="auto", max_features=1.0, random_state=cm.SEED, n_jobs=-1)
QUANTILE_GRID = [0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]


def operating_rows(frame):
    return frame.loc[(frame["prev_day_off"] == 0) & frame["hour"].between(8, 18)]


EXPERIMENTS = {
    "A_all_features": {"features": FEATURES, "fit_rows": lambda f: f, "directional": False},
    "B_power_features": {"features": POWER_FEATURES, "fit_rows": lambda f: f, "directional": False},
    "C_power_operating": {"features": POWER_FEATURES, "fit_rows": operating_rows, "directional": False},
    "C_power_operating_dir": {"features": POWER_FEATURES, "fit_rows": operating_rows, "directional": True},
}


def fit_score(exp, tr, frames):
    cfg = EXPERIMENTS[exp]
    fit = cfg["fit_rows"](tr)
    model = IsolationForest(**IF_PARAMS).fit(fit[cfg["features"]])
    out = []
    for f in [fit] + frames:
        s = -model.decision_function(f[cfg["features"]])
        if cfg["directional"]:  # 전력이 평소(학습 중앙값)보다 낮은 쪽의 '이상'은 피크 위험이 아님
            low = f["lag_1"].to_numpy() < np.median(fit["lag_1"])
            s = np.where(low, s.min() - 1.0, s)
        out.append(s)
    return out  # [train_score, *frame_scores]


rows, chosen = [], {}
for exp in EXPERIMENTS:
    ys, fold_scores, fold_train = [], [], []
    for _, tr, va in cm.cv_folds(ds):
        s_tr, s_va = fit_score(exp, tr, [va])
        ys.append(va["label"].to_numpy()); fold_scores.append(s_va); fold_train.append(s_tr)
    y = np.concatenate(ys)
    best = None
    for q in QUANTILE_GRID:  # fold별 train score의 (1-q) 분위수를 경계로 -> 이상 여부
        pred = np.concatenate([(s >= np.quantile(t, 1 - q)).astype(float) for s, t in zip(fold_scores, fold_train)])
        f1 = cm.evaluate(y, pred, 0.5)["F1"]
        if best is None or f1 > best[1]:
            best = (q, f1)
    s_all = np.concatenate(fold_scores)
    rows.append({"experiment": exp, "cv_PR-AUC": average_precision_score(y, s_all),
                 "cv_ROC-AUC": cm.roc_auc_score(y, s_all), "best_quantile": best[0], "cv_F1": best[1]})
    print(f"[CV] {exp:<24} PR-AUC {rows[-1]['cv_PR-AUC']:.4f} | F1 {best[1]:.4f} @ 상위 {best[0]:.0%}")

cv_table = pd.DataFrame(rows).sort_values(["cv_F1", "cv_PR-AUC"], ascending=False)
cv_table.to_csv(cm.out("detail", "isolation_forest_cv.csv"), index=False)
best_exp, best_q = cv_table.iloc[0]["experiment"], float(cv_table.iloc[0]["best_quantile"])
print(f"[선택] {best_exp}, 경계 = train score 상위 {best_q:.0%}")

# 최종: 9/1 이전 전체로 학습 -> test 1회 평가 (+ 기존 방식 A도 같은 절차로 기록)
for exp, name in [(best_exp, "isolation_forest"), ("A_all_features", "isolation_forest_A_all")]:
    q = best_q if exp == best_exp else float(cv_table.set_index("experiment").loc[exp, "best_quantile"])
    s_tr, s_te = fit_score(exp, train_df, [test_df])
    cut = float(np.quantile(s_tr, 1 - q))
    m = cm.save_results(name, test_df, s_te, cut, score_col="anomaly_score",
                        extra={"method": f"{exp}, top {q:.0%} of train score",
                               "cv_PR-AUC": float(cv_table.set_index('experiment').loc[exp, 'cv_PR-AUC']),
                               "cv_F1": float(cv_table.set_index('experiment').loc[exp, 'cv_F1'])})
    print(f"\n[{name}] {cm.fmt(m)}")
    if name == "isolation_forest":
        flagged = test_df.loc[s_te >= cut]
        print(f"이상 판정 {len(flagged)}건 중 실제 피크 {int(flagged['label'].sum())}건, "
              f"판정 시각 hour 분포 {flagged['hour'].value_counts().sort_index().to_dict()}")
