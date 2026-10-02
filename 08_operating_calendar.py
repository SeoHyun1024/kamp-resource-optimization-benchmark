"""
08 가동 캘린더 (에이전트 M1)

실제 전력으로 날짜별 가동 여부와 날 유형을 판정해 휴무 캘린더를 정리한다.

판정 규칙 (데이터에서 기준값을 계산, 9/1 이전 데이터로만 정함)
    1) 가동일 기준: preprocessing.py의 OFF_DAY_MAX_POWER(60)를 그대로 쓴다 (피크 모델의 prev_day_off와 같은 기준).
       원본 일자의 일 최대전력이 비가동일 최대 41 / 가동일 최소 104로 나뉘므로, 기준이 그 사이에 있는지 assert로 확인한다.
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
    - 257일 중 115일은 다른 날짜의 15분 전력값을 복사한 증강일이다. 복사일은 전력 패턴이 원본 날짜의
      것이라 달력·생산 기록과 비교하면 왜곡된다(예: 2/11 공휴일 = 1/11 월요일 가동 패턴 복사).
      따라서 기준값은 원본 일자로만 정하고, 평가지표는 원본 / 증강 / 전체를 나눠 보고하며 원본을 주 결과로 쓴다.
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


def load_copied() -> dict:
    import json
    try:
        with open(pp.out("prep", "copied_days.json"), encoding="utf-8") as f:
            return {pd.Timestamp(k): pd.Timestamp(v) for k, v in json.load(f).items()}
    except FileNotFoundError:
        return {}


def thresholds(df: pd.DataFrame, copied: dict) -> tuple[float, float]:
    """가동일 기준은 preprocessing.OFF_DAY_MAX_POWER(60)로 통일하고, 9/1 이전 '원본 일자'의 일 최대전력 분포로
    그 기준이 비가동일·가동일 사이의 간격 안에 있는지 확인한다. 시간 단위 가동 기준은 비가동일 최대값(41)."""
    p = df.loc[df.index < pp.TEST_START, "power"]
    dm = p.groupby(p.index.normalize()).max()
    dm = dm.loc[~dm.index.isin(list(copied))]
    day_max = np.sort(dm.to_numpy())
    i = int(np.argmax(np.diff(day_max)))
    day_thr = float(pp.OFF_DAY_MAX_POWER)
    assert day_max[i] < day_thr < day_max[i + 1], (
        f"가동일 기준 {day_thr}이 비가동일 최대 {day_max[i]}와 가동일 최소 {day_max[i + 1]} 사이에 있지 않음")
    hour_thr = float(day_max[i])  # 비가동일의 최대값을 넘으면 가동 중
    return float(day_thr), hour_thr, float(day_max[i]), float(day_max[i + 1])


def build_calendar(df: pd.DataFrame, day_thr: float, hour_thr: float, copied: dict) -> pd.DataFrame:
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
    # 증강 복제일 표시: 복제일의 날 유형은 원본 날짜의 패턴이므로 평가·M3 패턴 생성 시 분리한다
    cal["is_copied_day"] = cal.index.isin(list(copied)).astype(int)
    cal["copied_from"] = [copied[d].date() if d in copied else "" for d in cal.index]
    return cal


def _metrics(cal: pd.DataFrame) -> dict:
    agree = (cal["operating"] == cal["production_operating"])
    hol, wk = cal["calendar_type"] == "공휴일", cal["calendar_type"] == "평일"
    sat, sun = cal["dow"] == "토", cal["dow"] == "일"
    return {
        "days": len(cal),
        "operating_days": int(cal["operating"].sum()),
        "non_operating_days": int((cal["operating"] == 0).sum()),
        "production_agreement": round(float(agree.mean()), 4),
        "production_mismatch_days": int((~agree).sum()),
        "power_on_production_zero": int(((cal["operating"] == 1) & (cal["production_operating"] == 0)).sum()),
        "power_off_production_pos": int(((cal["operating"] == 0) & (cal["production_operating"] == 1)).sum()),
        "holidays": int(hol.sum()), "holiday_operating": int((hol & (cal["operating"] == 1)).sum()),
        "saturdays": int(sat.sum()), "saturday_operating": int((sat & (cal["operating"] == 1)).sum()),
        "sundays": int(sun.sum()), "sunday_operating": int((sun & (cal["operating"] == 1)).sum()),
        "weekdays": int(wk.sum()), "weekday_non_operating": int((wk & (cal["operating"] == 0)).sum()),
        "calendar_mismatch_days": int((cal["calendar_says_off"] == cal["operating"]).sum()),
        "calendar_mismatch_rate": round(float((cal["calendar_says_off"] == cal["operating"]).mean()), 4),
    }


def evaluate(cal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """원본(주 결과) / 증강 / 전체로 나눠 평가지표를 계산한다."""
    groups = {"원본": cal[cal["is_copied_day"] == 0], "증강": cal[cal["is_copied_day"] == 1], "전체": cal}
    table = pd.DataFrame({k: _metrics(v) for k, v in groups.items()})
    mism = cal.loc[cal["operating"] != cal["production_operating"]]
    return table, mism


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
    copied = load_copied()
    day_thr, hour_thr, lo, hi = thresholds(df, copied)
    print(f"[기준, 원본 일자 기준] 비가동일 최대 {lo:.0f}, 가동일 최소 {hi:.0f} -> 가동일 기준 {day_thr:.0f} (preprocessing.OFF_DAY_MAX_POWER), 시간 단위 가동 기준 > {hour_thr:.0f}")

    cal = build_calendar(df, day_thr, hour_thr, copied)
    table, mism = evaluate(cal)
    cal.to_csv(pp.out("prep", "operating_calendar.csv"), encoding="utf-8-sig")
    table.loc["day_threshold"] = day_thr
    table.loc["hour_threshold"] = hour_thr
    table.to_csv(pp.out("cmp", "operating_calendar_summary.csv"), encoding="utf-8-sig")
    mism.to_csv(pp.out("detail", "operating_calendar_mismatch.csv"), encoding="utf-8-sig")
    orig = cal[cal["is_copied_day"] == 0]
    type_table = pd.crosstab(orig["day_type"], orig["dow"]).reindex(index=TYPES, columns=DOW_KO, fill_value=0)
    type_table.to_csv(pp.out("cmp", "operating_calendar_types.csv"), encoding="utf-8-sig")
    plot_calendar(cal, pp.out("fig", "operating_calendar.png"))

    t = table.astype(object)
    def fr(col, a, b):
        return f"{int(t.loc[a, col])}/{int(t.loc[b, col])}"
    view = pd.DataFrame({col: {
        "일수": int(t.loc["days", col]),
        "생산 기록 일치율": f"{float(t.loc['production_agreement', col]):.1%}",
        "생산 불일치일": int(t.loc["production_mismatch_days", col]),
        "공휴일 가동": fr(col, "holiday_operating", "holidays"),
        "토요일 가동": fr(col, "saturday_operating", "saturdays"),
        "일요일 가동": fr(col, "sunday_operating", "sundays"),
        "평일 비가동": fr(col, "weekday_non_operating", "weekdays"),
        "달력과 다른 날": f"{int(t.loc['calendar_mismatch_days', col])} ({float(t.loc['calendar_mismatch_rate', col]):.1%})",
    } for col in ["원본", "증강", "전체"]})
    print("\n=== 평가지표 (주 결과: 원본 일자) ===")
    print(view.to_string())
    print(f"\n=== 날 유형 x 요일 (원본 일자) ===\n{type_table.to_string()}")
    print(f"\n=== test 기간(9/1~9/14) 캘린더 ===")
    print(cal.loc[cal["split"] == "test", ["dow", "calendar_type", "day_type", "day_max", "on_hours", "first_on_hour", "last_on_hour"]].to_string())
    if len(mism):
        print(f"\n[생산 기록 불일치일 {len(mism)}일]\n" + mism[["dow", "calendar_type", "day_type", "day_max", "on_hours", "production_sum", "is_copied_day", "copied_from"]].to_string())
