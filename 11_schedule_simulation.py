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
