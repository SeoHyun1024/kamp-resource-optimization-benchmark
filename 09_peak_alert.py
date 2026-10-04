"""
09 피크 경보 (에이전트 M2)

이미 학습·저장된 두 모델(RNN 회귀, RF 분류)의 예측을 묶어 시간별 경보 단계를 만든다. 재학습은 하지 않는다.
    입력: 4_model_details/rnn_oof.csv, random_forest_oof.csv   (CV 7·8월 OOF, 규칙 선택용)
          2_test_predictions/rnn_forecast.csv, random_forest_predictions.csv (test, 선택된 규칙을 한 번만 적용)
          5_models/rnn_meta.json, random_forest_meta.json   (RNN cutoff, RF threshold: 각 스크립트가 OOF로 고른 값)
          1_preprocessing/peak_dataset.csv                (근거 문구용 feature)
          1_preprocessing/operating_calendar.csv          (08 실행 결과, 없으면 생략)

[1] 규칙 선택 (CV OOF만 사용, test는 보지 않음)
    후보 규칙: RNN 단독(예측 >= cutoff) / RF 단독(확률 >= threshold) / AND / OR
    사전에 정한 기준
      주의: F2 최대 (놓치지 않기, Recall 우선)
      확정: 주의에 포함되는 규칙 중 F0.5 최대 (조치 대상, Precision 우선)
      동률 처리: 1위와의 차이를 날짜 단위 부트스트랩(2000회)으로 구해 95% 구간이 0을 포함하면
                 "차이 없음"으로 보고, 그중 단순한 규칙(모델 1개 > 2개)을 고른다
    fold(7월·8월)별 지표도 함께 저장해 한 달에만 좋은 규칙인지 확인한다
[2] test 적용: 선택된 규칙을 9/1~9/14에 한 번 적용해 경보 단계·근거 문구·전환 시각 안내를 만든다
    * 전환 시각(13·15·16·17시)은 헛경보가 몰리는 시각이라 transition_hour 플래그와 확인 안내를 붙인다
    * test의 다른 후보 성능은 참고로만 출력한다 (선택에 쓰지 않음)

출력: 3_comparison/peak_alert_rule_selection.csv (OOF 후보별 지표·부트스트랩 구간·fold별 지표)
      3_comparison/peak_alert_summary.csv        (test 경보 단계별 성능)
      2_test_predictions/peak_alerts.csv, 4_model_details/peak_alert_daily.csv
      5_models/peak_alert_meta.json              (선택된 규칙, M5 보고서용)
      6_figures/peak_alert_timeline.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import preprocessing as pp

TRANSITION_HOURS = [13, 15, 16, 17]
LEVELS = ["정상", "주의", "확정"]


def load_meta(name: str) -> dict:
    with open(pp.out("model", f"{name}_meta.json"), encoding="utf-8") as f:
        return json.load(f)


def reason(row) -> str:
    parts = []
    if row["same_hour_2w_max"] >= row["peak_threshold"]:
        parts.append(f"2주 내 같은 시각 최대 {row['same_hour_2w_max']:.0f}")
    if row["q60_lag1"] >= row["peak_threshold"] - 10:
        parts.append(f"직전 15분 {row['q60_lag1']:.0f}")
    if row["intra_hour_slope"] > 0:
        parts.append(f"직전 1시간 상승 +{row['intra_hour_slope']:.0f}")
    elif row["intra_hour_slope"] < 0:
        parts.append(f"직전 1시간 하락 {row['intra_hour_slope']:.0f}")
    return ", ".join(parts)


def metrics(y, pred, days) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    return {"alerts": tp + fp, "TP": tp, "FP": fp, "FN": fn, "Precision": prec, "Recall": rec,
            "F1": fbeta(tp, fp, fn, 1), "F2": fbeta(tp, fp, fn, 2), "F0.5": fbeta(tp, fp, fn, 0.5),
            "cost_5FN_FP": 5 * fn + fp, "alerts_per_day": (tp + fp) / days, "FP_per_day": fp / days}


def fbeta(tp, fp, fn, beta) -> float:
    b2 = beta ** 2
    d = (1 + b2) * tp + b2 * fn + fp
    return (1 + b2) * tp / d if d else 0.0


# ---------------- [1] 규칙 선택 (OOF) ----------------
RULES = ["RNN 단독", "RF 단독", "AND", "OR"]        # 단순한 순서 (모델 1개 -> 2개)
N_MODELS = {"RNN 단독": 1, "RF 단독": 1, "AND": 2, "OR": 2}
SUBSET_OF = {"RNN 단독": {"RNN 단독", "OR"}, "RF 단독": {"RF 단독", "OR"}, "AND": set(RULES), "OR": {"OR"}}
CRITERIA = {"주의": "F2", "확정": "F0.5"}
N_BOOT, BOOT_SEED = 2000, 42


def apply_rule(rule: str, rnn_alert, rf_alert):
    r, f = np.asarray(rnn_alert).astype(bool), np.asarray(rf_alert).astype(bool)
    return {"RNN 단독": r, "RF 단독": f, "AND": r & f, "OR": r | f}[rule].astype(int)


def _counts_by_day(y, pred, day_codes, n_days):
    tp = np.bincount(day_codes, (pred == 1) & (y == 1), n_days)
    fp = np.bincount(day_codes, (pred == 1) & (y == 0), n_days)
    fn = np.bincount(day_codes, (pred == 0) & (y == 1), n_days)
    return np.stack([tp, fp, fn])                      # (3, n_days)


def boot_diff(y, pred_best, pred_c, days, beta) -> tuple[float, float]:
    """날짜 단위 부트스트랩: F_beta(best) - F_beta(candidate)의 95% 구간."""
    codes, uniq = pd.factorize(pd.DatetimeIndex(days).normalize())
    n = len(uniq)
    cb, cc = _counts_by_day(y, pred_best, codes, n), _counts_by_day(y, pred_c, codes, n)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    sb, sc = cb[:, idx].sum(axis=2), cc[:, idx].sum(axis=2)     # (3, N_BOOT)
    d = fbeta_vec(sb[0], sb[1], sb[2], beta) - fbeta_vec(sc[0], sc[1], sc[2], beta)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def fbeta_vec(tp, fp, fn, beta):
    b2 = beta ** 2
    d = (1 + b2) * tp + b2 * fn + fp
    return np.where(d > 0, (1 + b2) * tp / np.maximum(d, 1), 0.0)


def load_oof(cutoff: float, rf_thr: float) -> pd.DataFrame:
    rnn = pd.read_csv(pp.out("detail", "rnn_oof.csv"), parse_dates=["Date"]).set_index("Date")
    rf = pd.read_csv(pp.out("detail", "random_forest_oof.csv"), parse_dates=["Date"]).set_index("Date")
    idx = rnn.index.intersection(rf.index)
    if len(idx) < len(rnn) or len(idx) < len(rf):
        print(f"[경고] OOF 시각 불일치: RNN {len(rnn)} / RF {len(rf)} / 공통 {len(idx)}")
    assert (rnn.loc[idx, "label"].to_numpy() == rf.loc[idx, "label"].to_numpy()).all(), "RNN·RF OOF label 불일치"
    if "predicted_label" in rnn and ((rnn["forecast"] >= cutoff).astype(int) != rnn["predicted_label"]).any():
        print("[경고] rnn_oof.csv의 predicted_label이 rnn_meta.json의 cutoff와 맞지 않음 -> 01_rnn.py를 끝까지 다시 실행하세요")
    o = pd.DataFrame({"fold": rnn.loc[idx, "fold"].astype(str), "label": rf.loc[idx, "label"].astype(int),
                      "rnn_alert": (rnn.loc[idx, "forecast"] >= cutoff).astype(int),
                      "rf_alert": (rf.loc[idx, "score"] >= rf_thr).astype(int)}, index=idx)
    return o


def select_rules(o: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    y, days = o["label"].to_numpy(), o.index
    n_days = days.normalize().nunique()
    preds = {r: apply_rule(r, o["rnn_alert"], o["rf_alert"]) for r in RULES}
    rows, chosen = [], {}
    for stage, crit in CRITERIA.items():
        beta = {"F2": 2, "F0.5": 0.5}[crit]
        cands = RULES if stage == "주의" else [r for r in RULES if chosen["주의"] in SUBSET_OF[r]]
        m = {r: metrics(y, preds[r], n_days) for r in cands}
        best = max(cands, key=lambda r: (m[r][crit], -N_MODELS[r]))
        tied = []
        for r in cands:
            lo, hi = (0.0, 0.0) if r == best else boot_diff(y, preds[best], preds[r], days, beta)
            if lo <= 0 <= hi:
                tied.append(r)
            row = {"stage": stage, "criterion": crit, "rule": r, "n_models": N_MODELS[r], **m[r],
                   "diff_vs_best": m[best][crit] - m[r][crit], "diff_CI_low": lo, "diff_CI_high": hi}
            for fold, g in o.groupby("fold"):
                fm = metrics(g["label"].to_numpy(), preds[r][o["fold"].to_numpy() == fold], g.index.normalize().nunique())
                row[f"{fold}_{crit}"] = fm[crit]; row[f"{fold}_Recall"] = fm["Recall"]
                row[f"{fold}_Precision"] = fm["Precision"]; row[f"{fold}_FP"] = fm["FP"]; row[f"{fold}_FN"] = fm["FN"]
            rows.append(row)
        pick = min(tied, key=lambda r: (N_MODELS[r], -m[r][crit]))
        chosen[stage] = pick
        for row in rows:
            if row["stage"] == stage:
                row["best_point"] = row["rule"] == best
                row["no_sig_diff"] = row["rule"] in tied
                row["selected"] = row["rule"] == pick
    return chosen, pd.DataFrame(rows)


def thresholds() -> dict:
    rnn_meta, rf_meta = load_meta("rnn"), load_meta("random_forest")
    return {"rnn_cutoff": float(rnn_meta["peak_cutoff"]), "rf_threshold": float(rf_meta["classification_threshold"]),
            "peak_threshold": float(rf_meta["peak_threshold"])}


# ---------------- [2] test 적용 ----------------
def build_alerts(cfg: dict, chosen: dict) -> pd.DataFrame:
    cutoff, rf_thr, peak_thr = cfg["rnn_cutoff"], cfg["rf_threshold"], cfg["peak_threshold"]

    rnn = pd.read_csv(pp.out("pred", "rnn_forecast.csv"), parse_dates=["Date"]).set_index("Date")
    rf = pd.read_csv(pp.out("pred", "random_forest_predictions.csv"), parse_dates=["Date"]).set_index("Date")
    ds = pd.read_csv(pp.out("prep", "peak_dataset.csv"), parse_dates=["Date"]).set_index("Date")
    idx = rf.index.intersection(rnn.index)

    a = pd.DataFrame(index=idx)
    a.index.name = "Date"
    a["hour"] = idx.hour
    a["actual_power"] = rf.loc[idx, "actual_power"]
    a["actual_peak"] = rf.loc[idx, "actual_label"].astype(int)
    a["rnn_forecast"] = rnn.loc[idx, "forecast"].round(1)
    a["rf_probability"] = rf.loc[idx, "risk_probability"].round(3)
    a["rnn_alert"] = (a["rnn_forecast"] >= cutoff).astype(int)
    a["rf_alert"] = (a["rf_probability"] >= rf_thr).astype(int)
    caution = apply_rule(chosen["주의"], a["rnn_alert"], a["rf_alert"])
    confirm = apply_rule(chosen["확정"], a["rnn_alert"], a["rf_alert"]) & caution   # 확정은 항상 주의에 포함
    a["level"] = np.select([confirm == 1, caution == 1], ["확정", "주의"], "정상")
    a["model_disagree"] = (a["rf_alert"] != a["rnn_alert"]).astype(int)       # 두 모델 판단이 갈린 시간
    a["transition_hour"] = a["hour"].isin(TRANSITION_HOURS).astype(int)

    feats = ds.loc[idx, ["lag_1", "q60_lag1", "intra_hour_slope", "same_hour_2w_max"]]
    a = a.join(feats)
    a["peak_threshold"] = peak_thr
    a["reason"] = [reason(r) if lv != "정상" else "" for lv, (_, r) in zip(a["level"], a.iterrows())]
    a["note"] = np.where((a["level"] != "정상") & a["transition_hour"].eq(1),
                         "전환 시각: 헛경보가 잦음. 15분 값이 170 아래로 내려오면 해제 검토", "")

    cal_path = pp.out("prep", "operating_calendar.csv")
    if os.path.exists(cal_path):
        cal = pd.read_csv(cal_path, parse_dates=["date"]).set_index("date")["day_type"]
        a["day_type"] = cal.reindex(idx.normalize()).to_numpy()
    return a.drop(columns="peak_threshold")


def plot_timeline(a: pd.DataFrame, cfg: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 3.8), dpi=160)
    ax.plot(a.index, a["actual_power"], color="#52514e", lw=1.2, label="실제 전력")
    ax.axhline(cfg["peak_threshold"], color="#898781", lw=0.8, ls=(0, (4, 3)))
    ax.text(a.index[-1], cfg["peak_threshold"] + 3, f"피크 기준 {cfg['peak_threshold']:.0f}", fontsize=8, color="#52514e", ha="right")
    for lv, c, y in [("주의", "#eda100", 228), ("확정", "#e34948", 236)]:
        t = a.index[a["level"].eq(lv)]
        ax.scatter(t, [y] * len(t), marker="|", s=60, color=c, label=f"{lv} 경보")
    miss = a.index[a["actual_peak"].eq(1) & a["level"].eq("정상")]
    ax.scatter(miss, a.loc[miss, "actual_power"], s=28, facecolor="none", edgecolor="#e34948", lw=1.2, label="놓친 피크")
    ax.set_ylim(0, 245); ax.set_ylabel("kW", fontsize=8)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("test 기간(9/1~9/14) 시간별 경보 단계", fontsize=10, loc="left")
    plt.tight_layout(); plt.savefig(path, facecolor="white"); plt.close()


if __name__ == "__main__":
    try:
        from matplotlib import font_manager as fm
        names = [x.name for x in fm.fontManager.ttflist]
        for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"]:
            if f in names:
                plt.rcParams["font.family"] = f; break
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    cfg = thresholds()
    print(f"[기준] RNN 예측 >= {cfg['rnn_cutoff']:.0f} | RF 확률 >= {cfg['rf_threshold']} | 피크 {cfg['peak_threshold']:.0f}")

    # [1] OOF로 규칙 선택
    o = load_oof(cfg["rnn_cutoff"], cfg["rf_threshold"])
    chosen, sel = select_rules(o)
    sel.to_csv(pp.out("cmp", "peak_alert_rule_selection.csv"), index=False, encoding="utf-8-sig")
    folds = sorted(o["fold"].unique())
    print(f"\n=== [1] 규칙 선택 (CV OOF {o.index.normalize().nunique()}일, 피크 {int(o['label'].sum())}건, test 미사용) ===")
    for stage, crit in CRITERIA.items():
        s = sel[sel["stage"] == stage]
        cols = ["rule", "Precision", "Recall", crit, "FP", "FN", "FP_per_day", "diff_CI_low", "diff_CI_high"] + \
               [f"{f}_{crit}" for f in folds] + ["no_sig_diff", "selected"]
        print(f"\n[{stage}] 기준 {crit} 최대, 1위와 차이 없으면 단순한 규칙")
        print(s[cols].round(3).to_string(index=False))
    print(f"\n-> 선택: 주의 = {chosen['주의']}, 확정 = {chosen['확정']} (주의에 포함)")
    if chosen["주의"] == chosen["확정"]:
        print("   두 단계 규칙이 같다: OOF에서 두 번째 확인이 유의한 이득을 주지 못해 경보가 1단계로 운영된다")

    # [2] test에 한 번 적용
    a = build_alerts(cfg, chosen)
    days = a.index.normalize().nunique()
    y = a["actual_peak"].to_numpy()
    warn = a["level"].isin(["주의", "확정"]).astype(int).to_numpy()
    conf = a["level"].eq("확정").astype(int).to_numpy()
    rows = {f"주의 이상 ({chosen['주의']})": metrics(y, warn, days), f"확정 ({chosen['확정']})": metrics(y, conf, days)}
    summary = pd.DataFrame(rows).T
    tr = a["transition_hour"].eq(1).to_numpy()
    summary["FP_transition_hours"] = [int((warn.astype(bool) & tr & (y == 0)).sum()), int((conf.astype(bool) & tr & (y == 0)).sum())]
    ref = pd.DataFrame({f"참고: {r}": metrics(y, apply_rule(r, a["rnn_alert"], a["rf_alert"]), days) for r in RULES}).T

    a.to_csv(pp.out("pred", "peak_alerts.csv"), encoding="utf-8-sig")
    pd.concat([summary, ref]).to_csv(pp.out("cmp", "peak_alert_summary.csv"), encoding="utf-8-sig")
    daily = a.assign(day=a.index.date).groupby("day").agg(
        day_type=("day_type", "first") if "day_type" in a else ("hour", "size"),
        peaks=("actual_peak", "sum"), caution=("level", lambda s: int((s == "주의").sum())),
        confirmed=("level", lambda s: int((s == "확정").sum())),
        missed=("actual_peak", lambda s: int(((s == 1) & (a.loc[s.index, "level"] == "정상")).sum())))
    daily.to_csv(pp.out("detail", "peak_alert_daily.csv"), encoding="utf-8-sig")
    plot_timeline(a, cfg, pp.out("fig", "peak_alert_timeline.png"))
    meta = {"script": "09_peak_alert.py", **cfg, "rules": chosen, "criteria": CRITERIA,
            "selection_data": f"CV OOF {folds} (복제일 제외), test 미사용",
            "tie_rule": f"1위와 차이의 날짜 부트스트랩({N_BOOT}회) 95% 구간이 0 포함 -> 단순한 규칙"}
    with open(pp.out("model", "peak_alert_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n=== [2] test 적용 ({days}일, 피크 {int(y.sum())}건) ===")
    show_cols = ["alerts", "TP", "FP", "FN", "Precision", "Recall", "F1", "F2", "F0.5", "FP_per_day"]
    print(summary[show_cols + ["FP_transition_hours"]].round(3).to_string())
    print("\n(참고, 선택에 쓰지 않음) test 후보별 성능")
    print(ref[show_cols].round(3).to_string())
    print(f"\n두 모델 판단이 갈린 시간: {int(a['model_disagree'].sum())}건")
    print(f"\n=== 날짜별 경보 ===\n{daily.to_string()}")
    ex = a[a["level"] != "정상"].head(5)
    print("\n=== 경보 예시 (근거 문구) ===")
    for t, r in ex.iterrows():
        print(f"{t:%m-%d %H시} [{r['level']}] RNN {r['rnn_forecast']:.0f} / RF {r['rf_probability']:.2f} | {r['reason']} {('| ' + r['note']) if r['note'] else ''}")
