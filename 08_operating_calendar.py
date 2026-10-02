"""
08 가동 캘린더 (에이전트 M1)

실제 전력으로 날짜별 가동 여부와 날 유형을 판정해 휴무 캘린더를 정리한다.

판정 규칙 (데이터에서 기준값을 계산, 9/1 이전 데이터로만 정함)
    1) 가동일 기준: 일 최대전력을 정렬했을 때 가장 큰 간격의 가운데 값
       (현재 데이터: 비가동일 최대 41, 가동일 최소 104 -> 기준 72.5)
    2) 시간 단위 가동 기준: 비가동일 전력의 최대값(41)을 넘으면 그 시간은 가동 중
    3) 날 유형
       - 비가동     : 일 최대전력 < 가동일 기준
       - 종일 가동  : 24시간 모두 가동 중
       - 가동 시작일: 하루 중 가동이 시작됨 (0시는 꺼져 있고 이후 켜짐)
       - 가동 종료일: 하루 중 가동이 끝남 (23시는 꺼져 있음)
       - 중간 정지  : 0시·23시는 가동 중이지만 하루 중 꺼진 시간이 있음
       - 부분 가동  : 시작과 종료가 같은 날

평가 (판정 기준과 독립적인 기록과 비교)
    - 생산 기록 일치율: 전력 기준 가동 여부 vs 생산량 > 0 인 날
    - 달력과의 불일치: 공휴일 가동, 주말 가동, 평일 비가동 일수
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import preprocessing as pp

HOLIDAYS = pd.to_datetime(["2021-01-01", "2021-02-11", "2021-02-12", "2021-03-01",
                           "2021-05-05", "2021-05-19", "2021-08-16"])
DOW_KO = ["월", "화", "수", "목", "금", "토", "일"]
TYPES = ["종일 가동", "가동 시작일", "가동 종료일", "중간 정지", "부분 가동", "비가동"]


def thresholds(df: pd.DataFrame) -> tuple[float, float]:
    """9/1 이전 데이터로 가동일 기준(일 최대전력 간격의 가운데)과 시간 단위 가동 기준을 정한다."""
    p = df.loc[df.index < pp.TEST_START, "power"]
    day_max = np.sort(p.groupby(p.index.normalize()).max().to_numpy())
    i = int(np.argmax(np.diff(day_max)))
    day_thr = (day_max[i] + day_max[i + 1]) / 2
    hour_thr = float(day_max[i])  # 비가동일의 최대값을 넘으면 가동 중
    return float(day_thr), hour_thr, float(day_max[i]), float(day_max[i + 1])


def build_calendar(df: pd.DataFrame, day_thr: float, hour_thr: float) -> pd.DataFrame:
    on = (df["power"] > hour_thr).astype(int)
    date = df.index.normalize()
    g = df.assign(on=on, hour=df.index.hour).groupby(date)
    cal = pd.DataFrame({
        "day_max": g["power"].max(),
        "on_hours": g["on"].sum(),
        "on_at_0h": g.apply(lambda x: int(x.loc[x["hour"] == 0, "on"].iloc[0])),
        "on_at_23h": g.apply(lambda x: int(x.loc[x["hour"] == 23, "on"].iloc[0])),
        "first_on_hour": g.apply(lambda x: x.loc[x["on"] == 1, "hour"].min()),
        "last_on_hour": g.apply(lambda x: x.loc[x["on"] == 1, "hour"].max()),
        "production_sum": g["생산량"].sum(),
        "production_hours": g["생산량"].apply(lambda s: int((s > 0).sum())),
    })
    cal.index.name = "date"
    cal["operating"] = (cal["day_max"] >= day_thr).astype(int)
    start = (cal["on_at_0h"] == 0) & (cal["on_hours"] > 0)
    end = (cal["on_at_23h"] == 0) & (cal["on_hours"] > 0)
    cal["day_type"] = np.select(
        [cal["operating"] == 0, start & end, start, end, cal["on_hours"] < 24],
        ["비가동", "부분 가동", "가동 시작일", "가동 종료일", "중간 정지"], default="종일 가동")
    cal["dow"] = [DOW_KO[d] for d in cal.index.dayofweek]
    cal["calendar_type"] = np.where(cal.index.isin(HOLIDAYS), "공휴일",
                                    np.where(cal.index.dayofweek >= 5, "주말", "평일"))
    cal["calendar_says_off"] = (cal["calendar_type"] != "평일").astype(int)
    cal["production_operating"] = (cal["production_sum"] > 0).astype(int)
    cal["split"] = np.where(cal.index >= pp.TEST_START, "test", "train")
    # 증강 복제일 표시: 복제일의 날 유형은 원본 날짜의 패턴이므로 M3 패턴 생성 시 제외한다
    try:
        import json
        with open(pp.out("prep", "copied_days.json"), encoding="utf-8") as f:
            copied = {pd.Timestamp(k): pd.Timestamp(v) for k, v in json.load(f).items()}
    except FileNotFoundError:
        copied = {}
    cal["is_copied_day"] = cal.index.isin(list(copied)).astype(int)
    cal["copied_from"] = [copied[d].date() if d in copied else "" for d in cal.index]
    return cal


def evaluate(cal: pd.DataFrame) -> dict:
    agree = (cal["operating"] == cal["production_operating"])
    mism = cal.loc[~agree]
    res = {
        "days": len(cal),
        "operating_days": int(cal["operating"].sum()),
        "non_operating_days": int((cal["operating"] == 0).sum()),
        "production_agreement": float(agree.mean()),
        "power_on_production_zero": int(((cal["operating"] == 1) & (cal["production_operating"] == 0)).sum()),
        "power_off_production_pos": int(((cal["operating"] == 0) & (cal["production_operating"] == 1)).sum()),
        "holiday_operating": int(((cal["calendar_type"] == "공휴일") & (cal["operating"] == 1)).sum()),
        "holidays": int((cal["calendar_type"] == "공휴일").sum()),
        "saturday_operating": int(((cal["dow"] == "토") & (cal["operating"] == 1)).sum()),
        "saturdays": int((cal["dow"] == "토").sum()),
        "sunday_operating": int(((cal["dow"] == "일") & (cal["operating"] == 1)).sum()),
        "sundays": int((cal["dow"] == "일").sum()),
        "weekday_non_operating": int(((cal["calendar_type"] == "평일") & (cal["operating"] == 0)).sum()),
        "calendar_mismatch_days": int((cal["calendar_says_off"] == cal["operating"]).sum()),
    }
    return res, mism


def plot_calendar(cal: pd.DataFrame, path: str) -> None:
    colors = {"종일 가동": "#2a78d6", "가동 시작일": "#86b6ef", "가동 종료일": "#1c5cab",
              "중간 정지": "#e87ba4", "부분 가동": "#eda100", "비가동": "#e6e5e1"}
    weeks = ((cal.index - cal.index[0].normalize() + pd.Timedelta(days=cal.index[0].dayofweek)).days // 7)
    fig, ax = plt.subplots(figsize=(13, 3.2), dpi=160)
    for (d, row), w in zip(cal.iterrows(), weeks):
        ax.add_patch(plt.Rectangle((w, 6 - d.dayofweek), 0.9, 0.9, color=colors[row["day_type"]], lw=0))
        if row["calendar_type"] == "공휴일":
            ax.text(w + 0.45, 6 - d.dayofweek + 0.45, "휴", ha="center", va="center", fontsize=6, color="white")
    ax.set_xlim(-0.2, weeks.max() + 1); ax.set_ylim(-0.2, 7.1)
    ax.set_yticks([6 - i + 0.45 for i in range(7)]); ax.set_yticklabels(DOW_KO, fontsize=8)
    month_starts = [i for i, d in enumerate(cal.index) if d.day == 1]
    ax.set_xticks([weeks[i] + 0.45 for i in month_starts]); ax.set_xticklabels([f"{cal.index[i].month}월" for i in month_starts], fontsize=8)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    present = [t for t in colors if (cal["day_type"] == t).any()]
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[t]) for t in present]
    ax.legend(handles, present, ncol=len(present), fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("가동 캘린더 (실제 전력 기준, '휴' = 달력상 공휴일)", fontsize=10, loc="left")
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

    df = pp.clean(pp.load_raw(), log=lambda *_: None)
    day_thr, hour_thr, lo, hi = thresholds(df)
    print(f"[기준] 비가동일 최대 {lo:.0f}, 가동일 최소 {hi:.0f} -> 가동일 기준 {day_thr:.1f}, 시간 단위 가동 기준 > {hour_thr:.0f}")

    cal = build_calendar(df, day_thr, hour_thr)
    res, mism = evaluate(cal)
    cal.to_csv(pp.out("prep", "operating_calendar.csv"), encoding="utf-8-sig")

    summary = pd.Series({**res, "day_threshold": day_thr, "hour_threshold": hour_thr})
    summary.to_csv(pp.out("cmp", "operating_calendar_summary.csv"), header=["value"], encoding="utf-8-sig")
    mism.to_csv(pp.out("detail", "operating_calendar_mismatch.csv"), encoding="utf-8-sig")
    type_table = pd.crosstab(cal["day_type"], cal["dow"]).reindex(index=TYPES, columns=DOW_KO, fill_value=0)
    type_table.to_csv(pp.out("cmp", "operating_calendar_types.csv"), encoding="utf-8-sig")
    plot_calendar(cal, pp.out("fig", "operating_calendar.png"))

    print(f"\n=== 가동 판정 ({res['days']}일) ===")
    print(f"가동 {res['operating_days']}일 / 비가동 {res['non_operating_days']}일")
    print(f"생산 기록 일치율 {res['production_agreement']:.1%} "
          f"(전력상 가동인데 생산 0: {res['power_on_production_zero']}일, 전력상 비가동인데 생산 있음: {res['power_off_production_pos']}일)")
    print(f"\n=== 달력과의 차이 ===")
    print(f"공휴일 {res['holidays']}일 중 가동 {res['holiday_operating']}일 | 토요일 {res['saturdays']}일 중 가동 {res['saturday_operating']}일 | "
          f"일요일 {res['sundays']}일 중 가동 {res['sunday_operating']}일 | 평일 비가동 {res['weekday_non_operating']}일")
    print(f"달력(평일=가동, 주말·공휴일=휴무)과 다른 날: {res['calendar_mismatch_days']}일 ({res['calendar_mismatch_days']/res['days']:.1%})")
    print(f"\n=== 날 유형 x 요일 ===\n{type_table.to_string()}")
    print(f"\n=== test 기간(9/1~9/14) 캘린더 ===")
    print(cal.loc[cal["split"] == "test", ["dow", "calendar_type", "day_type", "day_max", "on_hours", "first_on_hour", "last_on_hour"]].to_string())
    if len(mism):
        print(f"\n[생산 기록 불일치일 {len(mism)}일]\n" + mism[["dow", "calendar_type", "day_type", "day_max", "on_hours", "production_sum"]].to_string())
