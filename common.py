"""
02~06 모델 스크립트 공통 함수: CV 분할, 평가, threshold 선택, 결과 저장.

분할 원칙
    - CV (hyperparameter·threshold 선택): 확장창 2-fold
        fold 2021-07: train = 7/1 이전 전체, val = 7월
        fold 2021-08: train = 8/1 이전 전체, val = 8월
      두 fold의 validation 예측(out-of-fold)을 이어 붙여 PR-AUC/F1을 계산한다.
    - 최종 학습: 9/1 이전 전체 (기존 노트북은 7/31까지만 학습 -> 8월 정보를 버렸다)
    - Test: 2021-09-01 ~ 09-14, 최종 1회만 평가 (선택에 전혀 쓰지 않음)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

import preprocessing as pp

RESULTS_DIR = pp.RESULTS_DIR
out = pp.out
SEED = pp.SEED
THRESHOLD_GRID = pp.THRESHOLD_GRID

def _train_rows(ds: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    part = ds.loc[ds.index < end]
    if pp.DROP_COPIED_FROM_TRAIN:
        part = part.loc[part["is_copied_day"] == 0]
    return part


def cv_folds(ds: pd.DataFrame):
    """확장창 CV: [(fold명, train_df, val_df), ...]. 검증 구간도 복제일은 제외."""
    out = []
    for start, end in pp.CV_FOLDS:
        val = ds.loc[(ds.index >= start) & (ds.index < end)]
        val = val.loc[val["is_copied_day"] == 0]
        out.append((start.strftime("%Y-%m"), _train_rows(ds, start), val))
    return out


def final_split(ds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _train_rows(ds, pp.TEST_START), ds.loc[ds.index >= pp.TEST_START]


def evaluate(y_true, score, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, score),
        "PR-AUC": average_precision_score(y_true, score),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def best_threshold(y_true, score, grid=THRESHOLD_GRID) -> float:
    """F1 최대 threshold. 동률이면 가운데 값(0.5에 가까운 쪽)을 택해 경계값 과적합을 줄인다."""
    f1s = np.array([f1_score(y_true, (np.asarray(score) >= t).astype(int), zero_division=0) for t in grid])
    cands = grid[np.isclose(f1s, f1s.max())]
    return float(cands[np.argmin(np.abs(cands - 0.5))])


def save_results(model_name: str, test_df: pd.DataFrame, score, threshold: float,
                 extra: dict | None = None, score_col: str = "risk_probability", save_pred: bool = True) -> dict:
    """test 예측 -> results/2_test_predictions/<모델>_predictions.csv, 성능 -> results/3_comparison/model_comparison.csv(행 upsert)."""
    metrics = evaluate(test_df["label"], score, threshold)
    if save_pred:
        pd.DataFrame({
            "Date": test_df.index,
            "actual_label": test_df["label"].to_numpy(),
            "predicted_label": (np.asarray(score) >= threshold).astype(int),
            score_col: np.asarray(score),
            "actual_power": test_df["target_power"].to_numpy(),
        }).to_csv(out("pred", f"{model_name}_predictions.csv"), index=False)

    path = out("cmp", "model_comparison.csv")
    table = pd.read_csv(path, index_col="Model") if os.path.exists(path) else pd.DataFrame()
    row = pd.DataFrame([{**metrics, **(extra or {})}], index=pd.Index([model_name], name="Model"))
    table = pd.concat([table.drop(index=model_name, errors="ignore"), row])
    table.to_csv(path)
    return metrics


def fmt(m: dict) -> str:
    return (f"P {m['Precision']:.4f} | R {m['Recall']:.4f} | F1 {m['F1']:.4f} | "
            f"ROC {m['ROC-AUC']:.4f} | PR {m['PR-AUC']:.4f} | TP {m['TP']} FP {m['FP']} FN {m['FN']}")




def run_cv(ds: pd.DataFrame, grid: list[dict], fit_predict, tol: float = 0.005, log=print):
    """
    grid의 각 조합을 CV로 평가한다.
    fit_predict(params, train_df, val_df) -> (val_score 배열, best_iteration 또는 None)

    선택 규칙 (test 미사용):
      1) OOF PR-AUC(threshold와 무관한 순위 지표)가 최고값 - tol 이내인 후보만 남기고
      2) 그중 OOF 최적 F1이 가장 높은 조합을 고른다.
    반환: (결과표, 선택된 행 dict, 선택 조합의 OOF (y, score))
    """
    folds = cv_folds(ds)
    rows, oofs = [], []
    for i, params in enumerate(grid):
        ys, ss, iters, fold_pr = [], [], [], []
        for _, tr, va in folds:
            score, it = fit_predict(params, tr, va)
            ys.append(va["label"].to_numpy())
            ss.append(np.asarray(score, dtype=float))
            fold_pr.append(average_precision_score(ys[-1], ss[-1]))
            if it is not None:
                iters.append(it)
        y, s = np.concatenate(ys), np.concatenate(ss)
        thr = best_threshold(y, s)
        rows.append({
            **{k: ("None" if v is None else v) for k, v in params.items()},
            "cv_PR-AUC": average_precision_score(y, s),
            "cv_ROC-AUC": roc_auc_score(y, s),
            "cv_F1": evaluate(y, s, thr)["F1"],
            "cv_threshold": thr,
            "fold_PR-AUC": " / ".join(f"{v:.4f}" for v in fold_pr),
            "best_iter": int(np.mean(iters)) if iters else None,
        })
        oofs.append((y, s))
        log(f"  [{i + 1:>2}/{len(grid)}] {params} -> PR-AUC {rows[-1]['cv_PR-AUC']:.4f}, F1 {rows[-1]['cv_F1']:.4f} @ {thr}")
    table = pd.DataFrame(rows)
    ok = table["cv_PR-AUC"] >= table["cv_PR-AUC"].max() - tol
    best_i = table.loc[ok].sort_values(["cv_F1", "cv_PR-AUC"], ascending=False).index[0]
    return table.sort_values("cv_PR-AUC", ascending=False), table.loc[best_i].to_dict(), oofs[best_i], grid[best_i]



def save_oof(name: str, ds: pd.DataFrame, y, score) -> None:
    """선택된 조합의 CV out-of-fold 예측 저장 (07_compare.py 앙상블 threshold 선택용)."""
    idx = pd.DatetimeIndex(np.concatenate([va.index.to_numpy() for _, _, va in cv_folds(ds)]), name="Date")
    pd.DataFrame({"label": np.asarray(y), "score": np.asarray(score)}, index=idx).to_csv(out("detail", f"{name}_oof.csv"))
