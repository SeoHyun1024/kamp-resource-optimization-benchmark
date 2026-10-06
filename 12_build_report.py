"""
에이전트 M5: 추천 리포트 -> 웹 데이터

M1(가동 캘린더)·M2(피크 경보)·M3(하루 전 예측)·M4(일정 조정) 결과를 한 파일로 묶어
web/ 의 관리자 화면·근로자 시간표·챗봇이 읽게 한다. 웹은 서버가 없는 정적 페이지(GitHub Pages)라서
모든 값을 여기서 미리 계산해 둔다. 수치는 results/ 의 파일에서 읽고, 여기서 새로 만들지 않는다.

실행 순서: 08 -> 09 -> 10 -> 11 -> 15 -> 12.  (15_weekly_feedback.py 결과가 있으면 관리자 화면은 주간 피드백 10주로 만든다)   python 12_build_report.py   ->  web/data/agent_data.js
주간 계획 탭(선택): 13 -> 14 --web-examples 를 먼저 실행하면 plans/out/web/ 결과를 함께 넣는다. 없으면 그 탭은 비워 둔다.
14 --plan <CSV> --publish <이름> 으로 올린 내 계획도 같은 폴더에 등록되어 함께 들어간다(14가 12를 자동 재실행).
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


PLAN_WEB = os.path.join(HERE, "plans", "out", "web")


def load_plan_advice():
    """14_plan_advisor.py --web-examples 결과(plans/out/web/)를 웹용으로 묶는다. 없으면 None."""
    modes_path = os.path.join(PLAN_WEB, "modes.json")
    if not os.path.exists(modes_path):
        print("[주의] plans/out/web/ 이 없어 주간 계획 탭을 비워 둡니다 (13 -> 14 --web-examples 실행 필요)")
        return None
    with open(modes_path, encoding="utf-8") as f:
        modes = json.load(f)
    weeks = {}
    for ex in modes["examples"]:
        folder = os.path.join(PLAN_WEB, ex["key"])
        with open(os.path.join(folder, "summary.json"), encoding="utf-8") as f:
            sm = json.load(f)
        with open(os.path.join(folder, "advice_hourly.csv"), encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        with open(os.path.join(folder, "moves.csv"), encoding="utf-8-sig", newline="") as f:
            moves = list(csv.DictReader(f))
        num = lambda v: None if v in ("", None) else round(float(v), 1)
        hours = {k: [num(r[k]) for r in rows] for k in
                 ["pred_before", "pred_after", "production_before", "production", "actual", "actual_after_est"]
                 if k in rows[0]}
        hours["on_before"] = [int(r["on_before"]) for r in rows]
        hours["on_after"] = [int(r["on"]) for r in rows]
        mode = modes["modes"]["ext" if ex["ext"] else "base"]
        weeks[ex["key"]] = {
            "start": ex["start"], "ext": ex["ext"], "summary": sm["summary"], "days": sm["days"],
            "group": ex.get("group", ex["key"].removesuffix("_ext")), "label": ex.get("label", ex["key"]),
            "source": ex.get("source", "example"),
            "allow_new": ex.get("allow_new", mode["allow_new"]), "max_move_share": ex.get("max_move_share", mode["max_move_share"]),
            "time": [f"{r['date']} {int(r['hour']):02d}" for r in rows], "hours": hours,
            "moves": [{"kind": m["kind"], "from": m["from"], "to": m["to"], "qty": float(m["qty"]),
                       "new_hours": m["new_hours"].split(";") if m["new_hours"] else []} for m in moves],
        }
    with open(os.path.join(RES, "5_models", "week_ahead_meta.json"), encoding="utf-8") as f:
        wm = json.load(f)
    editor_path = os.path.join(PLAN_WEB, "editor_index.json")
    editor = []
    if os.path.exists(editor_path) and os.path.exists(os.path.join(HERE, "web", "data", "plan_editor.js")):
        with open(editor_path, encoding="utf-8") as f:
            editor = json.load(f)
    return {"modes": modes["modes"], "weeks": weeks, "editor": editor,
            "model": {"test_mae": wm["test"]["MAE"], "test_daily_max_mae": wm["test"]["daily_max_MAE"],
                      "cv_mae": wm["cv_MAE"], "margin": wm["safety_margin_kw"], "trained_until": wm["trained_until"]}}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # 관리자 화면 데이터: 15_weekly_feedback.py 결과(주마다 다시 학습한 7/7~9/14 10주)가 있으면 그것을,
    # 없으면 11_schedule_adjust.py 결과(8/31까지 학습한 모델 하나로 본 9/1~9/14)를 쓴다.
    fb_path = os.path.join(RES, "4_model_details", "weekly_feedback.json")
    fb = None
    if os.path.exists(fb_path):
        with open(fb_path, encoding="utf-8") as f:
            fb = json.load(f)
        meta4 = fb["meta"]
        days = {d: [{"day_type": v["day_type"], "actual": a, "pred": p, "on": o, "m2": m}
                    for a, p, o, m in zip(v["actual"], v["pred"], v["on"], v["m2"])] for d, v in fb["days"].items()}
        scenarios = fb["scenarios"]
        print(f"[정보] 관리자 화면: 주간 피드백 {len(fb['weeks'])}주 ({min(days)} ~ {max(days)})")
    else:
        print("[주의] results/4_model_details/weekly_feedback.json 이 없어 11 결과(9/1~9/14)로 만듭니다. (python 15_weekly_feedback.py)")
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
        if fb:
            out_days[d]["on_actual"] = fb["days"][d]["on_actual"]

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
            "notes": ([
                f"{min(days)}~{max(days)}를 주마다 되돌려 본 시뮬레이션이며 실제 운영 결과가 아닙니다.",
                "각 주는 그 주 시작 전날까지의 데이터로 M3를 다시 학습해 예측하고, 일정 조정 효과는 같은 날의 실제 전력에 적용해 계산했습니다.",
                "이동 가능 비율·기본요금 단가는 데이터에 없는 운영 조건이므로 입력값입니다.",
            ] if fb else [
                "2021-09-01~09-14 test 구간을 되돌려 본 시뮬레이션이며 실제 운영 결과가 아닙니다.",
                "일정은 M3 하루 전 예측과 가동 계획으로 짜고, 효과는 같은 날의 실제 전력에 적용해 계산했습니다.",
                "이동 가능 비율·기본요금 단가는 데이터에 없는 운영 조건이므로 입력값입니다.",
            ]),
            "feedback": bool(fb),
            "excluded_days": fb["meta"]["excluded_days"] if fb else [],
        },
        "days": out_days,
        "scenarios": {k: {"share": s["share"], "margin": s["margin"], "summary": s["summary"], "days": s["days"]}
                      for k, s in scenarios.items()},
        "weeks": fb["weeks"] if fb else None,
        "hindsight": fb["hindsight"] if fb else None,
        "plan_advice": load_plan_advice(),
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
