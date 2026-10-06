"""
에이전트 M4: 일정 조정 (피크 시간 부하 이동 시 최대수요·기본요금 절감 계산)

M3(10_plan_regression.py)의 하루 전 예측과 가동 계획만으로 "내일" 일정을 짜고,
그 일정을 실제 전력(test 9/1~9/14)에 적용해 효과를 확인한다.

방법 (날짜별, 같은 날 안에서만 이동)
    1. 전출: 계획상 가동 시간 중 예측 + 안전 마진 >= 피크 기준(179)인 시간에서
       (178 + 안전 마진 초과분)만큼, 최대 '이동 가능 비율'까지 덜어낸다.
    2. 전입: 계획상 가동이면서 전출이 없는 시간 중, 전출 시간에서 ±6시간 안에 있고
       예측 부하가 낮은 곳부터 채운다.
       전입 후에도 (예측 + 마진)이 상한(178 - 여유 10)을 넘지 않게 한다.
    3. 하루 안에 다 옮기지 못하는 양은 옮기지 않는다 (전출량을 비례 축소).
    4. 검증: 같은 이동량을 실제 전력에 적용한다. 계획은 예측으로 세우고, 평가는 실제로 한다.

안전 마진: M3는 일 최대전력을 평균 7.9 kW 낮게 예측한다(README M3 해석). 그래서 예측이
    179에 못 미쳐도 실제로는 넘을 수 있는 시간을 잡으려고 예측에 마진을 더해서 본다.
이동 가능 비율·안전 마진은 데이터에 없는 운영 조건이라 격자로 계산해 두고 고른다.

표준 라이브러리만 사용한다 (pandas 불필요). 입력: 08~10번 결과. 실행: python 11_schedule_adjust.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

SHARES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]   # 이동 가능 비율
MARGINS = [0.0, 8.0, 15.0]                      # 안전 마진 (kW)
DEFAULT_SHARE, DEFAULT_MARGIN = 0.10, 8.0
DEST_BUFFER = 10.0                              # 전입 후 상한 여유 (kW)
MAX_SHIFT_H = 6                                 # 이동 거리 제한 (시간): 8시 부하를 0시로 보내는 식의 안을 막는다


def read_csv(*parts):
    with open(os.path.join(RES, *parts), encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_inputs():
    plan = read_csv("2_test_predictions", "plan_regression_predictions.csv")
    alerts = {r["Date"]: r for r in read_csv("2_test_predictions", "peak_alerts.csv")}
    with open(os.path.join(RES, "5_models", "peak_alert_meta.json"), encoding="utf-8") as f:
        threshold = float(json.load(f)["peak_threshold"])
    days: "OrderedDict[str, list]" = OrderedDict()
    for r in plan:
        a = alerts.get(r["Date"], {})
        days.setdefault(r["Date"][:10], []).append({
            "h": int(r["Date"][11:13]),
            "actual": float(r["actual_power"]),
            "pred": float(r["predicted_power"]),
            "on": int(r["planned_on"]),
            "day_type": r["day_type"],
            "m2": a.get("level", ""),
        })
    return days, threshold


def adjust_day(hours, threshold, share, margin):
    """한 날의 일정 조정. 전출·전입 이동 목록과 조정 후 전력(예측·실제)을 돌려준다."""
    cap_src = threshold - 1.0
    cap_dst = threshold - 1.0 - margin - DEST_BUFFER
    pred = [x["pred"] for x in hours]
    act = [x["actual"] for x in hours]
    on = [x["on"] for x in hours]

    out = [0.0] * 24
    for h in range(24):
        if on[h]:
            out[h] = min(max(0.0, pred[h] + margin - cap_src), share * pred[h])
    room = [max(0.0, cap_dst - pred[h]) if on[h] and out[h] == 0 else 0.0 for h in range(24)]

    transfers, left_room = [], room[:]
    placed_total = 0.0
    for src in sorted(range(24), key=lambda i: -out[i]):
        need = out[src]
        for dst in sorted(range(24), key=lambda i: -left_room[i]):
            if need <= 1e-9 or left_room[dst] <= 1e-9 or abs(dst - src) > MAX_SHIFT_H:
                continue
            kw = min(need, left_room[dst])
            transfers.append({"from": src, "to": dst, "kw": kw})
            need -= kw
            left_room[dst] -= kw
            placed_total += kw
        out[src] -= need                     # 옮기지 못한 양은 원래 시간에 남는다

    out_p = [0.0] * 24
    in_p = [0.0] * 24
    for t in transfers:
        out_p[t["from"]] += t["kw"]
        in_p[t["to"]] += t["kw"]
    pred_after = [pred[h] - out_p[h] + in_p[h] for h in range(24)]

    # 실제 전력에 적용: 전출은 실제 부하의 이동 가능 비율을 넘지 않게 하고, 같은 양만 전입
    scale = [min(1.0, share * act[h] / out_p[h]) if out_p[h] > 0 else 0.0 for h in range(24)]
    act_out = [out_p[h] * scale[h] for h in range(24)]
    act_in = [0.0] * 24
    for t in transfers:
        act_in[t["to"]] += t["kw"] * scale[t["from"]]
    act_after = [act[h] - act_out[h] + act_in[h] for h in range(24)]
    return {
        "transfers": [{**t, "kw": round(t["kw"], 1)} for t in transfers if t["kw"] >= 0.05],
        "out": [round(v, 1) for v in out_p], "in": [round(v, 1) for v in in_p],
        "pred_after": [round(v, 1) for v in pred_after],
        "actual_after": [round(v, 1) for v in act_after],
    }


def summarize(days, adjusted, threshold):
    before_hours = after_hours = 0
    before_max = after_max = 0.0
    reductions, remaining_flagged, remaining = [], 0, 0
    moved = 0.0
    for d, hours in days.items():
        adj = adjusted[d]
        act = [x["actual"] for x in hours]
        aft = adj["actual_after"]
        before_hours += sum(v >= threshold for v in act)
        after_hours += sum(v >= threshold for v in aft)
        before_max, after_max = max(before_max, max(act)), max(after_max, max(aft))
        if max(act) >= threshold:
            reductions.append(max(act) - max(aft))
        moved += sum(adj["out"])
        for x, v in zip(hours, aft):
            if v >= threshold:
                remaining += 1
                remaining_flagged += x["m2"] in ("주의", "확정")
    return {
        "peak_hours_before": before_hours, "peak_hours_after": after_hours,
        "period_max_before": round(before_max, 1), "period_max_after": round(after_max, 1),
        "avg_daily_max_cut_on_peak_days": round(sum(reductions) / len(reductions), 1) if reductions else 0.0,
        "peak_days": len(reductions), "moved_kw_hours": round(moved, 1),
        "remaining_peak_hours": remaining, "remaining_flagged_by_m2": remaining_flagged,
    }


def main():
    days, threshold = load_inputs()
    scenarios = OrderedDict()
    for share in SHARES:
        for margin in MARGINS:
            adjusted = OrderedDict((d, adjust_day(h, threshold, share, margin)) for d, h in days.items())
            scenarios[f"r{int(round(share * 100))}_m{int(margin)}"] = {
                "share": share, "margin": margin,
                "summary": summarize(days, adjusted, threshold), "days": adjusted,
            }

    default_key = f"r{int(round(DEFAULT_SHARE * 100))}_m{int(DEFAULT_MARGIN)}"

    # 1) 시간별 결과 (기본 시나리오)
    path = os.path.join(RES, "2_test_predictions", "schedule_adjust_hourly.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Date", "actual_power", "predicted_power", "planned_on", "m2_level",
                    "shift_out_kw", "shift_in_kw", "predicted_after", "actual_after", "action"])
        for d, hours in days.items():
            adj = scenarios[default_key]["days"][d]
            for x in hours:
                h = x["h"]
                action = ("비가동" if not x["on"] else "전출" if adj["out"][h] > 0
                          else "전입" if adj["in"][h] > 0 else "유지")
                w.writerow([f"{d} {h:02d}:00:00", x["actual"], round(x["pred"], 1), x["on"], x["m2"],
                            adj["out"][h], adj["in"][h], adj["pred_after"][h], adj["actual_after"][h], action])

    # 2) 시나리오별 요약
    path = os.path.join(RES, "3_comparison", "schedule_adjust_summary.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        keys = list(next(iter(scenarios.values()))["summary"].keys())
        w.writerow(["scenario", "share", "margin_kw"] + keys)
        for k, s in scenarios.items():
            w.writerow([k, s["share"], s["margin"]] + [s["summary"][c] for c in keys])

    # 3) 메타 (M5 리포트·웹이 읽는다)
    meta = {"script": "11_schedule_adjust.py", "peak_threshold": threshold, "default": default_key,
            "shares": SHARES, "margins": MARGINS, "dest_buffer_kw": DEST_BUFFER, "max_shift_h": MAX_SHIFT_H,
            "note": "이동 가능 비율·안전 마진은 운영 조건이라 격자로 계산. 요금 단가는 데이터에 없어 입력값으로 처리."}
    with open(os.path.join(RES, "5_models", "schedule_adjust_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    s = scenarios[default_key]["summary"]
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(f"기본 시나리오 {default_key}")
    print(f"  피크 시간(>= {threshold:g}) {s['peak_hours_before']} -> {s['peak_hours_after']}")
    print(f"  기간 최대수요 {s['period_max_before']} -> {s['period_max_after']} kW")
    print(f"  피크가 있던 {s['peak_days']}일의 일 최대 평균 -{s['avg_daily_max_cut_on_peak_days']} kW")
    print(f"  남은 피크 {s['remaining_peak_hours']}시간 중 M2 경보 {s['remaining_flagged_by_m2']}시간")
    return days, scenarios, meta


if __name__ == "__main__":
    main()
