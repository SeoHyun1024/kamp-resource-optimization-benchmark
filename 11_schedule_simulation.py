"""
11 일정 조정 시뮬레이션 (에이전트 M4)

피크 경보가 난 시간의 부하 x%를 같은 날 다른 시간으로 옮겼을 때 최대수요와 기본요금이 얼마나 줄어드는지 계산한다.
재학습은 하지 않고 이미 저장된 M2·M3 예측을 그대로 쓴다. 놓친 피크는 못 깎고, 헛경보는 불필요한 조정이 된다.
    입력: 5_models/peak_alert_meta.json                       (09: 기준값, 선택된 주의·확정 조합)
          4_model_details/rnn_oof.csv, random_forest_oof.csv  (7·8월 OOF)
          4_model_details/plan_regression_oof.csv             (10: 7·8월 M3 예측, 현실 계획)
          2_test_predictions/peak_alerts.csv                  (09: test 경보)
          2_test_predictions/plan_regression_predictions.csv  (10: test M3 예측, 현실 계획)
          1_preprocessing/peak_dataset.csv, clean_hourly.csv  (lag_168, 월별 최대수요)

[1] 이동 규칙 (결과를 보기 전에 고정)
    깎는 쪽: 경보 시간의 부하를 x%(5·10·20%) 줄인다. 앞에서 받은 양이 있으면 받은 뒤 부하에 적용한다
    받는 쪽: 같은 날 현실 계획상 가동 시간 중 예상 부하가 낮은 순. 예상 부하 = max(M3 예측, 2주 내 같은 시각 최대)
             한 시간이 받는 양은 (피크 기준 - 1) - (예상 부하 + 이미 받은 양)과 예상 부하 * x 중 작은 값까지
             1시간 전 대상은 경보 시각 이후로만, 하루 전 대상은 경보가 없는 시간이면 앞뒤 모두
    그날 안에 못 옮긴 양은 깎지 않는다 (이동 실패로 집계). 총 부하는 유지된다
    완벽 예측(상한)은 경보와 받는 시간의 예상 부하를 모두 실제 전력으로 안다고 본다
    * 처음에는 예상 부하를 M3 예측만으로 두고 시간당 한도가 없었는데, 7·8월 OOF에서 M3가 낮게 본 시간으로
      부하가 몰려 최대수요가 커져(7월 222 -> 257) 위처럼 고쳤다. test는 보지 않았다
[2] 평가 순서: 7·8월 OOF -> test 9/1~9/14 한 번 (--oof-only 로 7·8월만 먼저 확인)
    * M3 cutoff는 7·8월 OOF F1 최대로 고르므로 7·8월의 M3 결과는 낙관적이다

출력: 3_comparison/schedule_simulation_summary.csv     (기간 x 대상 x x별 지표·절감액)
      2_test_predictions/schedule_adjustments.csv      (test 이동 기록, M5 입력)
      4_model_details/schedule_simulation_hourly.csv   (시간별 조정 전·후 전력)
      5_models/schedule_simulation_meta.json           (가정, 단가, 출처, M3 cutoff)
      6_figures/schedule_simulation.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import preprocessing as pp

X_GRID = [0.05, 0.10, 0.20]
POWER_STEP = 1.0   # 전력 값 단위(kW). 받는 시간은 피크 기준보다 한 단위 아래까지만 채운다 (피크는 기준 이상)
RATCHET_MONTHS = [12, 1, 2, 7, 8, 9]   # 요금적용전력 산정에 들어가는 달 (+ 당월), 한전 기본공급약관 제68조
BASE_RATE = 8320                       # 기본요금 단가 (원/kW), 산업용전력(을) 고압A 선택Ⅱ
BASE_RATE_RANGE = (6490, 9810)         # 산업용 고압A 최저(갑 선택Ⅰ)·최고(을 선택Ⅲ) 단가 (원/kW)
TARIFF_CONFIRMED = False               # 단가 자체는 요금표로 확인. 이 공장이 그 요금제인지는 추정이라 False
TARIFF_SOURCE = ("한국전력공사 전기요금표(산업용) https://cyber.kepco.co.kr/ckepco/front/jsp/CY/E/E/CYEEHP00103.jsp : "
                 "산업용전력(을) 고압A 선택Ⅱ 기본요금 8,320원/kW. 요금제는 추정: 데이터의 전기요금 109.8/167.2/191.6이 "
                 "이 요금제 최대부하 단가(2013.11.21 시행 109.3/166.7/191.1)보다 각각 0.5 높은 같은 묶음")
BILLING_SOURCE = ("한국전력공사 기본공급약관 제68조(요금적용전력의 결정) "
                  "https://cyber.kepco.co.kr/ckepco/front/jsp/CY/D/C/CYDCHP00108.jsp")
M3_CUTOFF_GRID = list(range(150, 180))
RULES = ["RNN 단독", "RF 단독", "AND", "OR"]                    # 09_peak_alert.py의 후보와 같음
CEILING = {"after": "완벽 예측 (1시간 전)", "any": "완벽 예측 (하루 전)"}   # 같은 방향 제약의 상한


# ---------------- [1] 이동 규칙 ----------------
def shift_load(power, expected, planned_on, alert, x, cap, direction):
    """하루치(시간순) 부하 이동. expected = 받는 시간의 예상 부하.
    반환: (조정 후 전력, [(깎는 위치, 받는 위치, kW)], 이동 실패 kW)"""
    adj = np.asarray(power, dtype=float).copy()
    expected = np.asarray(expected, dtype=float)
    planned_on, alert = np.asarray(planned_on, dtype=bool), np.asarray(alert, dtype=bool)
    received = np.zeros(len(adj))
    moves, unmoved = [], 0.0
    for h in np.flatnonzero(alert):
        left = x * adj[h]
        if direction == "after":      # 경보 시점에는 이후 경보를 모른다 -> 이후 가동 시간 전부가 후보
            cand = [j for j in range(h + 1, len(adj)) if planned_on[j]]
        else:                         # 하루 전 계획 -> 경보가 없는 가동 시간이면 앞뒤 모두
            cand = [j for j in range(len(adj)) if planned_on[j] and not alert[j]]
        for j in sorted(cand, key=lambda j: (expected[j], j)):
            if left <= 1e-9:
                break
            room = min(cap - (expected[j] + received[j]),      # 받아도 피크 기준을 넘지 않게
                       x * expected[j] - received[j])           # 시간당 한도: 예상 부하의 x%
            if room <= 1e-9:
                continue
            amt = min(left, room)
            adj[h] -= amt; adj[j] += amt; received[j] += amt; left -= amt
            moves.append((int(h), int(j), float(amt)))
        unmoved += max(left, 0.0)
    assert (expected + received <= np.maximum(expected, cap) + 1e-6).all(), "받는 시간이 상한을 넘음"
    assert (received <= x * np.maximum(expected, 0) + 1e-6).all(), "받는 시간이 시간당 한도를 넘음"
    return adj, moves, float(unmoved)


def simulate(frame: pd.DataFrame, alert, expected, x: float, direction: str, cap: float):
    """날짜별로 shift_load를 적용. 반환: (조정 후 전력, 이동 기록[from, to, kw], 이동 실패 kW)"""
    assert frame.index.is_monotonic_increasing, "frame은 시간순이어야 함"
    power, on = frame["power"].to_numpy(float), frame["planned_on"].to_numpy().astype(bool)
    alert, expected = np.asarray(alert, dtype=bool), np.asarray(expected, dtype=float)
    codes, _ = pd.factorize(frame.index.normalize())
    adj, rows, unmoved = power.copy(), [], 0.0
    for c in np.unique(codes):
        i = np.flatnonzero(codes == c)
        a, mv, um = shift_load(power[i], expected[i], on[i], alert[i], x, cap, direction)
        assert abs(a.sum() - power[i].sum()) < 1e-6, "하루 부하 합계가 달라짐"
        adj[i] = a; unmoved += um
        rows += [(frame.index[i[s]], frame.index[i[d]], kw) for s, d, kw in mv]
    return adj, pd.DataFrame(rows, columns=["from", "to", "kw"]), unmoved


# ---------------- [2] 지표·요금 ----------------
def metrics(power, adj, alert, n_days, peak, moved_kw, unmoved_kw) -> dict:
    power, adj, alert = np.asarray(power, float), np.asarray(adj, float), np.asarray(alert, bool)
    was_peak = power >= peak
    return {"max_before": float(power.max()), "max_after": float(adj.max()),
            "reduction_kw": float(power.max() - adj.max()),
            "peaks_before": int(was_peak.sum()), "peaks_after": int((adj >= peak).sum()),
            "missed_peaks": int((was_peak & ~alert).sum()),          # 경보가 없어 못 깎은 피크
            "new_peaks": int((~was_peak & (adj >= peak)).sum()),     # 받아서 새로 피크가 된 시간
            "adjustments": int(alert.sum()), "adjustments_per_day": float(alert.sum() / n_days),
            "unnecessary": int((alert & ~was_peak).sum()),           # 헛경보로 한 조정
            "moved_kw": float(moved_kw), "unmoved_kw": float(unmoved_kw)}


def billing_kw(monthly_max: dict, month: int) -> float:
    """요금적용전력: 당월과, 직전 12개월 중 12·1·2·7·8·9월분 최대수요 가운데 가장 큰 값.
    데이터가 2021-01부터라 같은 해의 앞선 달만 본다 (계약전력 30% 하한은 계약전력을 몰라 적용하지 않음)."""
    prev = [v for m, v in monthly_max.items() if m < month and m in RATCHET_MONTHS]
    return float(max([monthly_max[month]] + prev))


def savings(max_after: float, month: int, monthly_max: dict, other_days_max: float, rate: float) -> dict:
    """other_days_max: 그 달에서 시뮬레이션하지 않은 날(복제일 등)의 최대. 조정 후 월 최대의 하한이 된다."""
    before = float(monthly_max[month])
    after = float(max(max_after, other_days_max))
    b0, b1 = billing_kw(monthly_max, month), billing_kw({**monthly_max, month: after}, month)
    return {"month_max_before": before, "month_max_after": after,
            "saving_kw_month": before - after, "saving_won_month": (before - after) * rate,
            "saving_rate_month": (before - after) / before,
            "billing_kw_before": b0, "billing_kw_after": b1, "saving_won_12m": (b0 - b1) * rate}


# ---------------- [3] 입력·비교 대상 ----------------
def require(kind: str, name: str, script: str) -> str:
    path = pp.out(kind, name)
    if not os.path.exists(path):
        raise SystemExit(f"[중단] {path} 없음 -> {script} 를 먼저 실행하세요")
    return path


def _read(kind: str, name: str, script: str) -> pd.DataFrame:
    return pd.read_csv(require(kind, name, script), parse_dates=["Date"]).set_index("Date")


def build_frame(m3: pd.DataFrame, rnn_alert: pd.Series, rf_alert: pd.Series, hist: pd.DataFrame) -> pd.DataFrame:
    """네 입력의 공통 시각 중 24시간이 온전한 날만 남긴 시간별 표. hist = peak_dataset의 lag_168, same_hour_2w_max."""
    idx = m3.index.intersection(rnn_alert.index).intersection(rf_alert.index).intersection(hist.index).sort_values()
    if len(idx) < max(len(m3), len(rnn_alert), len(rf_alert)):
        print(f"[경고] 입력 시각 불일치: M3 {len(m3)} / RNN {len(rnn_alert)} / RF {len(rf_alert)} / 공통 {len(idx)}")
    f = pd.DataFrame({"power": m3.loc[idx, "actual_power"].astype(float),
                      "m3_forecast": m3.loc[idx, "predicted_power"].astype(float),
                      "planned_on": m3.loc[idx, "planned_on"].astype(int),
                      "rnn_alert": rnn_alert.loc[idx].astype(int), "rf_alert": rf_alert.loc[idx].astype(int),
                      "lag_168": hist.loc[idx, "lag_168"].astype(float)}, index=idx)
    f["expected"] = np.maximum(f["m3_forecast"], hist.loc[idx, "same_hour_2w_max"].astype(float))   # 받는 시간의 예상 부하
    n = f.groupby(f.index.normalize())["power"].transform("size")
    if (n != 24).any():
        dropped = sorted({d.strftime("%m-%d") for d in f.index[n != 24].normalize()})
        print(f"[경고] 24시간이 안 되는 날 제외: {dropped}")
        f = f.loc[n == 24]
    assert not f.isna().any().any(), "입력에 결측이 있음"
    return f


def load_frame(period: str, meta: dict) -> pd.DataFrame:
    hist = _read("prep", "peak_dataset.csv", "preprocessing.py")[["lag_168", "same_hour_2w_max"]]
    if period == "oof":
        m3 = _read("detail", "plan_regression_oof.csv", "10_plan_regression.py")
        rnn = _read("detail", "rnn_oof.csv", "01_rnn.py")["forecast"] >= meta["rnn_cutoff"]
        rf = _read("detail", "random_forest_oof.csv", "02_random_forest.py")["score"] >= meta["rf_threshold"]
    else:
        m3 = _read("pred", "plan_regression_predictions.csv", "10_plan_regression.py")
        al = _read("pred", "peak_alerts.csv", "09_peak_alert.py")      # 09가 test에 적용한 판정을 그대로 씀
        rnn, rf = al["rnn_alert"], al["rf_alert"]
    return build_frame(m3, rnn, rf, hist)


def _apply_rule(rule: str, r: np.ndarray, f: np.ndarray) -> np.ndarray:
    return {"RNN 단독": r, "RF 단독": f, "AND": r & f, "OR": r | f}[rule]


def alert_sources(frame: pd.DataFrame, rules: dict, m3_cutoff: float, peak: float) -> dict:
    """비교 대상별 (경보 배열, 방향, 받는 시간의 예상 부하). 방향 after = 경보 시각 이후로만 이동, any = 앞뒤 모두.
    완벽 예측은 받는 시간의 부하도 실제 전력으로 안다고 본다 (절감 상한)."""
    for stage in ("주의", "확정"):
        if rules.get(stage) not in RULES:
            raise ValueError(f"peak_alert_meta.json의 {stage} 규칙 '{rules.get(stage)}'은 {RULES} 중 하나여야 함")
    r, f = frame["rnn_alert"].to_numpy().astype(bool), frame["rf_alert"].to_numpy().astype(bool)
    power, exp = frame["power"].to_numpy(float), frame["expected"].to_numpy(float)
    actual = power >= peak
    caution = _apply_rule(rules["주의"], r, f)
    confirm = _apply_rule(rules["확정"], r, f) & caution              # 확정은 항상 주의에 포함
    return {CEILING["after"]: (actual, "after", power),
            CEILING["any"]: (actual, "any", power),
            f"M2 주의 ({rules['주의']})": (caution, "after", exp),
            f"M2 확정 ({rules['확정']})": (confirm, "after", exp),
            "참고: RNN 단독": (r, "after", exp),
            "M3 하루 전": (frame["m3_forecast"].to_numpy() >= m3_cutoff, "any", exp),
            "1주 전 같은 시각": (frame["lag_168"].to_numpy() >= peak, "any", exp)}


def select_m3_cutoff(frame: pd.DataFrame, peak: float) -> int:
    """M3 예측을 경보로 바꾸는 cutoff: OOF F1 최대, 같으면 가장 낮은 값."""
    y = (frame["power"] >= peak).astype(int)
    return max(M3_CUTOFF_GRID, key=lambda c: f1_score(y, (frame["m3_forecast"] >= c).astype(int), zero_division=0))


# ---------------- [4] 실행 ----------------
def run_period(frame: pd.DataFrame, label: str, sources: dict, peak: float, bill: dict):
    """한 기간의 대상 x x 전체 시뮬레이션. 반환: (요약, 시간별 조정 전·후, 이동 기록)"""
    cap = peak - POWER_STEP
    power = frame["power"].to_numpy(float)
    n_days = frame.index.normalize().nunique()
    base = metrics(power, power, np.zeros(len(power), bool), n_days, peak, 0.0, 0.0)
    rows = [{"period": label, "source": "조정 전", "direction": "-", "x": 0.0, **base, **savings(base["max_after"], **bill)}]
    hourly, moves_all = [], []
    for name, (alert, direction, expected) in sources.items():
        for x in X_GRID:
            adj, moves, unmoved = simulate(frame, alert, expected, x, direction, cap)
            m = metrics(power, adj, alert, n_days, peak, moves["kw"].sum(), unmoved)
            rows.append({"period": label, "source": name, "direction": direction, "x": x, **m, **savings(m["max_after"], **bill)})
            hourly.append(pd.DataFrame({"period": label, "source": name, "x": x, "Date": frame.index,
                                        "before": power, "after": adj, "alert": np.asarray(alert).astype(int)}))
            moves_all.append(moves.assign(period=label, source=name, x=x))
    summary = pd.DataFrame(rows)
    ceil = summary.set_index(["source", "x"])["reduction_kw"]

    def ratio(r):                       # 같은 방향 제약의 완벽 예측 대비 달성률
        top = ceil.get((CEILING.get(r["direction"]), r["x"]), np.nan)
        return r["reduction_kw"] / top if top and top > 0 else np.nan
    summary["achieved_ratio"] = summary.apply(ratio, axis=1)
    return summary, pd.concat(hourly, ignore_index=True), pd.concat(moves_all, ignore_index=True)


def bill_context(clean: pd.DataFrame, frame: pd.DataFrame, month: int) -> dict:
    monthly_max = clean.groupby(clean.index.month)["power"].max().to_dict()
    in_month = clean.loc[clean.index.month == month, "power"]
    other = in_month.loc[~in_month.index.normalize().isin(frame.index.normalize())]
    return {"month": month, "monthly_max": monthly_max,
            "other_days_max": float(other.max()) if len(other) else 0.0, "rate": BASE_RATE}


def plot(summary: pd.DataFrame, hourly: pd.DataFrame, peak: float, path: str) -> None:
    periods = list(summary["period"].unique())
    fig, axes = plt.subplots(1, len(periods) + 1, figsize=(6 * (len(periods) + 1), 5))
    for ax, p in zip(axes, periods):
        s = summary[(summary["period"] == p) & (summary["source"] != "조정 전")]
        piv = s.pivot(index="source", columns="x", values="reduction_kw").loc[s["source"].unique()]
        piv.columns = [f"x={c:.0%}" for c in piv.columns]
        piv.plot.barh(ax=ax)
        ax.set_title(f"{p} 최대수요 감소량 (kW)"); ax.set_ylabel(""); ax.invert_yaxis()
    last = periods[-1]
    h = hourly[(hourly["period"] == last) & (hourly["source"].str.startswith("M2 주의")) & (hourly["x"] == 0.10)].set_index("Date")
    day = h["before"].idxmax().normalize()                       # 그 기간 최대수요가 난 날
    d = h.loc[h.index.normalize() == day]
    axes[-1].step(d.index.hour, d["before"], where="mid", label="조정 전")
    axes[-1].step(d.index.hour, d["after"], where="mid", label="조정 후 (M2 주의, x=10%)")
    axes[-1].axhline(peak, color="gray", ls="--", lw=1, label=f"피크 기준 {peak:.0f}")
    axes[-1].set_title(f"{day:%m-%d} 시간별 전력"); axes[-1].set_xlabel("시"); axes[-1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main(oof_only: bool) -> None:
    try:
        from matplotlib import font_manager
        names = {f.name for f in font_manager.fontManager.ttflist}
        for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"]:
            if f in names:
                plt.rcParams["font.family"] = f; break
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    with open(require("model", "peak_alert_meta.json", "09_peak_alert.py"), encoding="utf-8") as fp:
        meta = json.load(fp)
    peak = float(meta["peak_threshold"])
    clean = pd.read_csv(require("prep", "clean_hourly.csv", "preprocessing.py"), parse_dates=["Date"]).set_index("Date")

    oof = load_frame("oof", meta)
    m3_cutoff = select_m3_cutoff(oof, peak)
    print(f"[기준] 피크 {peak:.0f} | x {X_GRID} | 받는 상한 {peak - POWER_STEP:.0f} | M3 cutoff {m3_cutoff} (7·8월 OOF F1 최대)")
    print(f"[조합] 주의 = {meta['rules']['주의']}, 확정 = {meta['rules']['확정']} | 기본요금 {BASE_RATE:,}원/kW")

    frames = [(f"2021-{m:02d}", m, oof.loc[oof.index.month == m]) for m in sorted(oof.index.month.unique())]
    if not oof_only:
        test = load_frame("test", meta)
        frames.append(("test", int(test.index.month[0]), test))
    parts = []
    for label, month, frame in frames:
        bill = bill_context(clean, frame, month)
        assert bill["monthly_max"][month] >= frame["power"].max(), "월 최대가 기간 최대보다 작음"
        parts.append(run_period(frame, label, alert_sources(frame, meta["rules"], m3_cutoff, peak), peak, bill))
    summary = pd.concat([p[0] for p in parts], ignore_index=True)
    hourly = pd.concat([p[1] for p in parts], ignore_index=True)
    moves = pd.concat([p[2] for p in parts], ignore_index=True)

    summary.to_csv(pp.out("cmp", "schedule_simulation_summary.csv"), index=False, encoding="utf-8-sig")
    hourly.to_csv(pp.out("detail", "schedule_simulation_hourly.csv"), index=False, encoding="utf-8-sig")
    plot(summary, hourly, peak, pp.out("fig", "schedule_simulation.png"))
    if not oof_only:
        moves[moves["period"] == "test"].drop(columns="period").to_csv(
            pp.out("pred", "schedule_adjustments.csv"), index=False, encoding="utf-8-sig")
    out_meta = {"script": "11_schedule_simulation.py", "peak_threshold": peak, "x_grid": X_GRID, "receive_cap": peak - POWER_STEP,
                "rules": meta["rules"], "rnn_cutoff": meta["rnn_cutoff"], "rf_threshold": meta["rf_threshold"],
                "m3_cutoff": m3_cutoff, "m3_cutoff_selection": "7·8월 OOF F1 최대 (7·8월의 M3 결과는 낙관적)",
                "shift_rule": "경보 시간 부하의 x%를 같은 날 현실 계획상 가동 시간 중 예상 부하가 낮은 순으로 이동. "
                              "예상 부하 = max(M3 예측, 2주 내 같은 시각 최대). 한 시간이 받는 양은 피크 기준 - 1까지의 여유와 "
                              "예상 부하의 x% 중 작은 값. 1시간 전 대상은 이후 시간만, 하루 전 대상은 경보 없는 시간 앞뒤 모두. "
                              "못 옮긴 양은 깎지 않음. 완벽 예측은 받는 시간의 부하도 실제 전력으로 앎",
                "rule_revision": "처음 규칙(예상 부하 = M3 예측, 시간당 한도 없음)은 7·8월 OOF에서 최대수요를 키워 수정함. test 미사용",
                "base_rate_won_per_kw": BASE_RATE, "base_rate_range": list(BASE_RATE_RANGE),
                "tariff_confirmed": TARIFF_CONFIRMED, "tariff_source": TARIFF_SOURCE, "billing_source": BILLING_SOURCE,
                "billing_rule": f"요금적용전력 = 당월과 직전 12개월 중 {RATCHET_MONTHS}월분 최대수요 중 최댓값 (데이터는 2021-01부터)",
                "periods": [label for label, _, _ in frames]}
    with open(pp.out("model", "schedule_simulation_meta.json"), "w", encoding="utf-8") as fp:
        json.dump(out_meta, fp, ensure_ascii=False, indent=2)

    pd.set_option("display.width", 250)
    show = ["source", "x", "max_after", "reduction_kw", "achieved_ratio", "peaks_after", "missed_peaks", "new_peaks",
            "adjustments_per_day", "unnecessary", "unmoved_kw", "saving_won_month", "saving_won_12m"]
    for label, _, frame in frames:
        s = summary[summary["period"] == label]
        b = s.iloc[0]
        print(f"\n=== {label} ({frame.index.normalize().nunique()}일, 피크 {b['peaks_before']}건, 최대 {b['max_before']:.0f} kW, "
              f"월 최대 {b['month_max_before']:.0f} kW, 요금적용전력 {b['billing_kw_before']:.0f} kW) ===")
        print(s.iloc[1:][show].round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-only", action="store_true", help="7·8월 OOF만 실행 (test는 건드리지 않음)")
    main(ap.parse_args().oof_only)
