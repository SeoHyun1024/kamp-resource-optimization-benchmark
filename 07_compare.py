"""
07 최종 비교: 전체 모델 test 성능 + 앙상블 + 불확실성(부트스트랩)

실행 순서: preprocessing.py -> 01_rnn.py, 02 ~ 06 -> 07_compare.py
    - 앙상블: RF·XGBoost·CatBoost 확률 평균. threshold는 세 모델 CV OOF 평균 확률로 선택(test 미사용)
    - RNN: results/2_test_predictions/rnn_forecast.csv(01_rnn.py 결과)가 있으면 같은 표에 포함
    - 불확실성: test 14일을 "일 단위 블록"으로 복원추출(2000회) -> F1 95% 구간
      (시간 단위로 뽑으면 자기상관 때문에 구간이 과소추정되므로 일 단위)
산출: results/3_comparison/final_comparison.csv
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

import common as cm

R = cm.RESULTS_DIR
N_BOOT = 2000
rng = np.random.default_rng(cm.SEED)


def load_pred(name: str) -> pd.DataFrame:
    file = "rnn_forecast.csv" if name == "rnn" else f"{name}_predictions.csv"
    return pd.read_csv(cm.out("pred", file), index_col="Date", parse_dates=True)


# --- 1) 앙상블 -----------------------------------------------------------------
members = ["random_forest", "xgboost", "catboost"]
oof = pd.concat({m: pd.read_csv(cm.out("detail", f"{m}_oof.csv"), index_col="Date", parse_dates=True)["score"] for m in members}, axis=1)
oof_label = pd.read_csv(cm.out("detail", "random_forest_oof.csv"), index_col="Date", parse_dates=True)["label"]
ens_thr = cm.best_threshold(oof_label, oof.mean(axis=1))
test_scores = pd.concat({m: load_pred(m)["risk_probability"] for m in members}, axis=1)
base = load_pred("random_forest")
test_df = pd.DataFrame({"label": base["actual_label"], "target_power": base["actual_power"]})
ens_cv = cm.evaluate(oof_label, oof.mean(axis=1), ens_thr)
cm.save_results("ensemble_rf_xgb_cb", test_df, test_scores.mean(axis=1).to_numpy(), ens_thr,
                extra={"method": "RF/XGB/CB 확률 평균", "cv_PR-AUC": ens_cv["PR-AUC"], "cv_F1": ens_cv["F1"]})
print(f"[앙상블] CV PR-AUC {ens_cv['PR-AUC']:.4f}, CV F1 {ens_cv['F1']:.4f} @ {ens_thr}")

# --- 2) RNN (01_rnn.py 결과가 있으면 포함) --------------------------------------------
rnn_path = cm.out("pred", "rnn_forecast.csv")
if os.path.exists(rnn_path):
    rnn = pd.read_csv(rnn_path, index_col="Date", parse_dates=True)
    if "predicted_label" in rnn.columns and rnn.index.equals(test_df.index):
        meta_path = cm.out("model", "rnn_meta.json")
        if os.path.exists(meta_path):                      # 01_rnn.py가 CV OOF로 고른 cutoff
            with open(meta_path, encoding="utf-8") as f:
                cutoff = float(json.load(f)["peak_cutoff"])
        else:                                              # meta가 없으면 예측 라벨에서 역산
            pos = rnn.loc[rnn["predicted_label"] == 1, "forecast"]
            cutoff = float(pos.min()) if len(pos) else float(rnn["forecast"].max() + 1)
        extra = {"method": f"01_rnn.py, 예측값 >= {cutoff:.0f}"}
        oof_path = cm.out("detail", "rnn_oof.csv")
        if os.path.exists(oof_path):                       # RNN CV 지표 (7·8월 OOF)
            oof = pd.read_csv(oof_path)
            extra["cv_F1"] = f1_score(oof["label"], (oof["forecast"] >= cutoff).astype(int))
            extra["cv_PR-AUC"] = average_precision_score(oof["label"], oof["forecast"])
        cm.save_results("rnn", test_df, rnn["forecast"].to_numpy(), cutoff, score_col="predicted_power",
                        extra=extra, save_pred=False)
        print(f"[RNN] rnn_forecast.csv 포함 (피크 판정 cutoff {cutoff:.0f})")
else:
    print("[RNN] results/2_test_predictions/rnn_forecast.csv 없음 -> 01_rnn.py 실행 후 다시 돌리면 표에 포함됩니다")

# --- 3) 일 단위 블록 부트스트랩 ------------------------------------------------------------
days = test_df.index.normalize()
day_rows = [np.where(days == d)[0] for d in days.unique()]
boot_idx = [np.concatenate([day_rows[i] for i in rng.integers(0, len(day_rows), len(day_rows))]) for _ in range(N_BOOT)]
y = test_df["label"].to_numpy()

table = pd.read_csv(cm.out("cmp", "model_comparison.csv"), index_col="Model")
ci = {}
for name in table.index:
    pred = load_pred(name)["predicted_label"].to_numpy()
    f = np.array([f1_score(y[b], pred[b], zero_division=0) for b in boot_idx])
    ci[name] = f"[{np.quantile(f, .025):.3f}, {np.quantile(f, .975):.3f}]"
final = table.assign(**{"F1_95%CI": pd.Series(ci)})
order = ["persistence", "persistence_lag168", "random_forest", "xgboost", "catboost", "ensemble_rf_xgb_cb",
         "rnn", "xgboost_reg", "lightgbm_reg", "isolation_forest", "isolation_forest_A_all"]
final = final.loc[[o for o in order if o in final.index]]
final.to_csv(cm.out("cmp", "final_comparison.csv"))

cols = [c for c in ["threshold", "Precision", "Recall", "F1", "F1_95%CI", "ROC-AUC", "PR-AUC", "cv_F1", "cv_PR-AUC"] if c in final]
pd.set_option("display.width", 220)
print(f"\n=== Test 2021-09-01~09-14 (피크 {int(y.sum())}건) ===")
print(final[cols].round(4).to_string())
print("\n[읽는 법] test는 14일·피크 47건이라 F1 95% 구간 폭이 약 ±0.15. 구간이 겹치는 모델 간 차이는 확정할 수 없다.")