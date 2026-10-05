"""
에이전트 M5: 추천 리포트 -> 웹 데이터

M1(가동 캘린더)·M2(피크 경보)·M3(하루 전 예측)·M4(일정 조정) 결과를 한 파일로 묶어
web/ 의 관리자 화면·근로자 시간표·챗봇이 읽게 한다. 웹은 서버가 없는 정적 페이지(GitHub Pages)라서
모든 값을 여기서 미리 계산해 둔다. 수치는 results/ 의 파일에서 읽고, 여기서 새로 만들지 않는다.

실행 순서: 08 -> 09 -> 10 -> 11 -> 12.   python 12_build_report.py   ->  web/data/agent_data.js
표준 라이브러리만 사용한다.
"""
from __future__ import annotations

import csv
import importlib
import json
import os
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "web", "data", "agent_data.js")


def read_csv(*parts):
    with open(os.path.join(RES, *parts), encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pick(rows, key, value):
    for r in rows:
        if r[key] == value:
            return r
    raise KeyError(f"{key}={value}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    m4 = importlib.import_module("11_schedule_adjust")
    days, scenarios, meta4 = m4.main()

    # M2 / M3 근거 수치 (README와 같은 파일에서 읽는다)
    alert = pick(read_csv("3_comparison", "peak_alert_summary.csv"), "", "주의 이상 (RF 단독)")
    confirm = pick(read_csv("3_comparison", "peak_alert_summary.csv"), "", "확정 (AND)")
    cmp3 = pick(read_csv("3_comparison", "plan_regression_comparison.csv"), "Model", "B''_계획예측 / lightgbm")
    folds = read_csv("3_comparison", "plan_regression_fold_comparison.csv")
    bias = {r["fold"]: float(r["daily_max_bias"]) for r in folds if r["Model"] == "B''_계획예측 / lightgbm"}
    bias["test"] = float(cmp3["daily_max_bias"])
    cal = {r["date"]: r for r in read_csv("1_preprocessing", "operating_calendar.csv")}

    out_days = {}
    for d, hours in days.items():
        y, m, dd = map(int, d.split("-"))
        out_days[d] = {
            "dow": date(y, m, dd).weekday(),            # 0=월 ... 6=일
            "day_type": hours[0]["day_type"],
            "calendar_type": cal.get(d, {}).get("calendar_type", ""),
            "actual": [x["actual"] for x in hours],
            "pred": [round(x["pred"], 1) for x in hours],
            "on": [x["on"] for x in hours],
            "m2": [x["m2"] for x in hours],
        }

    payload = {
        "meta": {
            "title": "KAMP 자원 최적화 에이전트",
            "period": [min(days), max(days)],
            "peak_threshold": meta4["peak_threshold"],
            "default": meta4["default"],
            "shares": meta4["shares"],
            "margins": meta4["margins"],
            "dest_buffer_kw": meta4["dest_buffer_kw"],
            "max_shift_h": meta4["max_shift_h"],
            "margin_basis": {k: round(v, 1) for k, v in bias.items()},
            "models": {
                "m3_mae": round(float(cmp3["MAE"]), 2),
                "m3_daily_max_mae": round(float(cmp3["daily_max_MAE"]), 2),
                "m2_alert": {"precision": round(float(alert["Precision"]), 3), "recall": round(float(alert["Recall"]), 3)},
                "m2_confirm": {"precision": round(float(confirm["Precision"]), 3), "recall": round(float(confirm["Recall"]), 3)},
            },
            "notes": [
                "2021-09-01~09-14 test 구간을 되돌려 본 시뮬레이션이며 실제 운영 결과가 아닙니다.",
                "일정은 M3 하루 전 예측과 가동 계획으로 짜고, 효과는 같은 날의 실제 전력에 적용해 계산했습니다.",
                "이동 가능 비율·기본요금 단가는 데이터에 없는 운영 조건이므로 입력값입니다.",
            ],
        },
        "days": out_days,
        "scenarios": {k: {"share": s["share"], "margin": s["margin"], "summary": s["summary"], "days": s["days"]}
                      for k, s in scenarios.items()},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # .js 로 저장: file:// 로 열어도, GitHub Pages 에서도 fetch 없이 <script> 한 줄로 읽힌다
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("window.AGENT_DATA = ")
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")
    print(f"저장: {os.path.relpath(OUT, HERE)} ({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
