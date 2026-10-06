"""
14 주간 계획 수정 제안 (에이전트 M4-W)

다음 1주일 계획(시간별 가동 여부 + 생산량) CSV를 넣으면
    1) 13_week_ahead.py 모델로 그 주의 시간별 전력을 예측하고
    2) 날마다의 최대전력을 낮추도록 생산량을 옮기는 안을 찾아
    3) 수정된 계획 CSV와 설명을 낸다.

목표: 일 최대전력 낮추기
    기본요금은 최대수요전력으로 매겨지므로, 피크 기준(179)을 다 없애지 못해도 최대를 몇 kW 낮추는 것이 곧 절감 근거다.
    탐색 점수 = sum_날짜 softmax_최대(예측) + 0.2 x 위험 초과량
      - softmax_최대: tau * log(sum(exp(예측 / tau))), tau = 2 kW. 최대와 비슷한 시간이 여럿이면 함께 낮춰야 점수가 준다.
      - 위험 초과량: sum(max(0, 예측 + 안전 마진 - 178)). 피크 기준 근처 시간을 우선하도록 보조로 둔다.

수단 1 (기본): 생산량 이동
    - 주간 총 생산량은 그대로 두고 옮기기만 한다.
    - 1단계: 같은 날 ±window시간 안에서 옮긴다. 2단계: 다른 날의 가동 시간으로 옮긴다(--same-day-only 로 끔).
      13 모델은 시간별 생산량보다 '그날 총 생산량'에 더 크게 반응하므로, 실제 효과는 주로 2단계에서 나온다.
    - 한 시간에서 옮길 수 있는 양: 원래 생산량의 --max-move-share(기본 50%)까지, 한 번에 25%씩.
    - 받는 시간의 생산량은 최근 8주 가동 시간 생산량의 95% 분위수(설비 한도 대용)를 넘지 않는다.

수단 2 (선택, --allow-new-hours): 가동 시간대 연장
    - 지금 비가동인 시간을 켜서 생산량을 받는다. 단 아래 두 조건을 모두 만족하는 시간만:
        a) 같은 날 바로 옆 시간이 가동 중 (가동 구간을 늘리는 것만, 따로 떨어진 가동은 만들지 않음)
        b) 과거(원본 일자)에 그 요일·시각에 가동한 비율이 5% 이상 (예: 토요일 낮, 월요일 새벽 = 선례가 있는 운영)
    - 평일 낮을 끄는 안은 만들지 않는다. 데이터에 선례가 없어 모델 예측을 믿을 수 없기 때문이다.
    - 결과에는 '실험적 제안'으로 표시한다.

실행 예
    python 14_plan_advisor.py --template 2021-09-15                  # 평소 일정 템플릿 -> plans/plan_2021-09-15.csv
    (CSV를 열어 on / production 수정)
    python 14_plan_advisor.py --plan plans/plan_2021-09-15.csv       # -> plans/out/plan_2021-09-15/
    python 14_plan_advisor.py --plan plans/plan_2021-09-15.csv --allow-new-hours --out plans/out/plan_2021-09-15_ext

    백테스트: 계획 시작일이 데이터 안이면(예: 2021-09-08) 9/1 이전 자료로 학습한 모델과 계획 시작 전 전력만 써서
    예측하고 실제 전력과 함께 보여준다.
    python 14_plan_advisor.py --template 2021-09-08 --source actual  # 실제로 실행된 계획을 템플릿으로
    python 14_plan_advisor.py --web-examples                         # 웹 화면용 예시 4개 -> plans/out/web/ (그 다음 12 실행)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

import plan_tools as pt
import preprocessing as pp

HERE = os.path.dirname(os.path.abspath(__file__))
PLANS_DIR = os.path.join(HERE, "plans")
STEP_SHARE = 0.25          # 한 번에 옮기는 양 = 원래 생산량의 25%
DEST_HEADROOM = 3.0        # 받는 시간은 (예측 + 마진)이 178 - 3 아래일 때만
TAU = 2.0                  # softmax 최대의 온도 (kW)
EXCESS_WEIGHT = 0.2        # 위험 초과량 보조 가중치
NEW_HOUR_PENALTY = 1.0     # 비가동 시간을 켜는 안은 이만큼 손해로 보고 비교 (점수 단위 kW)
CROSS_DAY_PENALTY = 0.2    # 다른 날로 옮기는 안
PRECEDENT_MIN_SHARE = 0.05 # 비가동 시간을 켤 수 있는 최소 과거 가동 비율
MAX_ITER = 150
MIN_GAIN = 0.3             # 이동 하나가 점수를 이만큼 이상 줄여야 채택 (의미 없는 이동 방지)
DAY_STEP = 0.10            # 다른 날로 한 번에 옮기는 양 = 보내는 날 하루 생산량의 10%
DEFAULT_PRICE = 8000       # 기본요금 예시 단가 (원/kW·월, 웹 화면 기본값과 같음)


def fmt_t(t: pd.Timestamp) -> str:
    return f"{t:%m/%d}({pt.DOW_KO[t.dayofweek]}) {t:%H}시"


# =============================================================================
# 준비
# =============================================================================
class Context:
    def __init__(self, plan_start: pd.Timestamp, model_override: str | None = None):
        self.df, self.copied, self.hour_thr, self.peak_thr = pt.load_history()
        self.executed = pt.executed_plan(self.df, self.copied, self.hour_thr)
        self.meta = pt.load_meta(pp.out("model", "week_ahead_meta.json"))
        self.data_end = self.df.index.max()
        self.backtest = plan_start <= self.data_end
        if model_override:
            model_file = model_override
        elif self.backtest:
            model_file = pp.out("model", self.meta["backtest_model"]["file"])
            trained_until = pd.Timestamp(self.meta["backtest_model"]["trained_until"])
            if plan_start <= trained_until:
                print(f"[주의] 백테스트 모델은 {trained_until:%m/%d}까지 학습했습니다. 그 이전 주는 학습에 쓴 날이라 결과가 낙관적입니다.")
        else:
            model_file = pp.out("model", "week_ahead.joblib")
        self.model = joblib.load(model_file)
        self.model_file = os.path.basename(model_file)
        self.feats = self.meta["features"]
        self.margin = float(self.meta["safety_margin_kw"])
        self.cap = self.peak_thr - 1.0
        self.history = self.df.loc[self.df.index < plan_start]           # 예측에는 계획 시작 직전까지의 전력만
        hist_plan = self.executed.loc[self.executed.index < plan_start]
        recent = hist_plan.loc[hist_plan.index >= plan_start - pd.Timedelta(weeks=8)]
        self.prod_cap = float(recent.loc[recent["on"] == 1, "production"].quantile(0.95))
        self.share = pt.weekday_share(hist_plan, self.copied, plan_start)  # 요일·시각별 과거 가동 비율


def check_window(ctx: Context, plan: pd.DataFrame) -> None:
    start, end = plan.index.min(), plan.index.max()
    if start.hour != 0 or len(plan) % 24:
        raise ValueError("계획은 0시부터 23시까지 하루 단위로 넣어 주세요.")
    if end - start >= pd.Timedelta(hours=pt.HORIZON):
        raise ValueError(f"계획은 최대 7일(168시간)입니다. 현재 {len(plan)}시간.")
    last_known = ctx.history.index.max()
    if start - last_known > pd.Timedelta(hours=1):
        raise ValueError(f"계획 시작 {start:%Y-%m-%d} 직전까지의 실제 전력이 필요합니다. "
                         f"데이터는 {last_known:%Y-%m-%d %H}시에서 끝납니다.")
    if ctx.history.index.min() > start - pd.Timedelta(hours=pt.WEEK_LAGS[-1]):
        raise ValueError("계획 시작 전 3주치 전력이 필요합니다.")


def fill_production(ctx: Context, plan: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    usual = pt.usual_plan(ctx.df, ctx.executed, ctx.copied, plan.index.min(), plan.index)
    missing = plan["production"].isna()
    plan = plan.assign(production=plan["production"].fillna(usual["production"]))
    plan.loc[plan["on"] == 0, "production"] = 0.0
    return plan, int(missing.sum())


# =============================================================================
# 예측과 점수
# =============================================================================
def predict_many(ctx: Context, base: pd.DataFrame, plans: list[pd.DataFrame]) -> np.ndarray:
    frames = [pd.concat([base, pt.plan_features(p)], axis=1)[ctx.feats] for p in plans]
    return ctx.model.predict(pd.concat(frames, axis=0)).reshape(len(plans), len(base))


def predict_days(ctx: Context, base: pd.DataFrame, plans: list[pd.DataFrame], days: list):
    """후보 계획마다 영향받는 날(days)만 다시 예측한다. 계획 feature는 날 단위로 계산되고 날 경계에서만
    앞뒤 시각을 보므로, 앞뒤 하루를 붙여 계산한 뒤 해당 날만 잘라 쓴다 (전체 예측과 같은 값인지 assert로 확인)."""
    t = base.index
    day = t.normalize()
    keep = day.isin(days)
    ctx_mask = day.isin([d + pd.Timedelta(days=k) for d in days for k in (-1, 0, 1)])
    rows = [pd.concat([base.loc[keep], pt.plan_features(p.loc[ctx_mask]).loc[t[keep]]], axis=1)[ctx.feats]
            for p in plans]
    pred = ctx.model.predict(pd.concat(rows, axis=0))
    return pred.reshape(len(plans), int(keep.sum())), np.flatnonzero(keep)


def score(ctx: Context, pred: np.ndarray) -> np.ndarray:
    """pred: (..., 시간) — 시간 수는 24의 배수(하루 단위). 날마다 softmax 최대의 합 + 위험 초과량 보조."""
    p = pred.reshape(pred.shape[:-1] + (-1, 24))
    m = p.max(axis=-1, keepdims=True)
    smax = (m[..., 0] + TAU * np.log(np.exp((p - m) / TAU).sum(axis=-1))).sum(axis=-1)
    exc = np.maximum(0.0, pred + ctx.margin - ctx.cap).sum(axis=-1)
    return smax + EXCESS_WEIGHT * exc


# =============================================================================
# 탐색
# =============================================================================
def search(ctx: Context, plan: pd.DataFrame, base: pd.DataFrame, same_day_only: bool, allow_new: bool,
           max_move_share: float, window: int):
    """탐욕 탐색.
    1단계(같은 날): 날 최대에 가까운 시간의 생산량 25%를 같은 날 ±window시간의 한가한 가동 시간으로.
    2단계(다른 날): 최대가 높은 날의 하루 생산량 DAY_STEP(10%)를 최대가 낮은 날로. 보내는 날은 피크 시간대(예측 상위)
      위주로 덜어내고, 받는 날은 예측이 낮은 가동 시간부터 채운다. --allow-new-hours면 받는 날의 가동 구간 끝에
      선례가 있는 비가동 시간을 붙여 늘린다. 날마다 하루 생산량의 max_move_share까지만 내보낸다.
    매번 후보를 모델로 다시 예측해 점수(score)가 가장 많이 줄어드는 안 하나를 고른다."""
    cur = plan.copy()
    orig_prod = plan["production"].to_numpy().copy()
    pred = predict_many(ctx, base, [cur])[0]
    moves = []
    t = plan.index
    day = t.normalize()
    days_all = sorted(set(day))
    day_idx = {d0: np.flatnonzero(day == d0) for d0 in days_all}
    day_total0 = {d0: orig_prod[day_idx[d0]].sum() for d0 in days_all}
    day_out = {d0: 0.0 for d0 in days_all}
    hour_out = np.zeros(len(t))
    precedent = ctx.share.to_numpy()[t.dayofweek, t.hour] >= PRECEDENT_MIN_SHARE

    def evaluate(cands):
        """cands: [(새 계획, 영향받는 날 목록, 이동 정보, 벌점)] -> 가장 좋은 후보를 적용하면 True."""
        nonlocal cur, pred
        if not cands:
            return False
        days = sorted({d0 for c in cands for d0 in c[1]})
        preds, rows = predict_days(ctx, base, [c[0] for c in cands], days)
        gain = score(ctx, pred[rows]) - score(ctx, preds) - np.array([c[3] for c in cands])
        k = int(np.argmax(gain))
        if gain[k] <= MIN_GAIN:
            return False
        cur = cands[k][0]
        pred = pred.copy()
        pred[rows] = preds[k]
        info = cands[k][2]
        hour_out[:] += info.pop("_hour_out")
        if info["kind"] == "day":
            day_out[info["_from_day"]] += info["qty"]
        info.pop("_from_day", None)
        moves.append(info)
        return True

    # ---- 1단계: 같은 날 안에서 시간 이동 ----
    for d0 in days_all:
        idx = day_idx[d0]
        for _ in range(MAX_ITER):
            on, prod = cur["on"].to_numpy(), cur["production"].to_numpy()
            srcs = sorted([i for i in idx if on[i] == 1 and prod[i] >= 1
                           and hour_out[i] < max_move_share * orig_prod[i] - 1e-9], key=lambda i: -pred[i])[:6]
            cands = []
            for s in srcs:
                qty = min(STEP_SHARE * orig_prod[s], max_move_share * orig_prod[s] - hour_out[s], prod[s])
                for d in idx:
                    if (qty < 1 or d == s or on[d] == 0 or abs(d - s) > window or pred[d] >= pred[s] - 5
                            or pred[d] + ctx.margin >= ctx.cap - DEST_HEADROOM or prod[d] + qty > ctx.prod_cap):
                        continue
                    p = cur.copy()
                    p.iloc[s, p.columns.get_loc("production")] -= qty
                    p.iloc[d, p.columns.get_loc("production")] += qty
                    ho = np.zeros(len(t)); ho[s] = qty
                    cands.append((p, [d0], {"kind": "hour", "from": t[s], "to": t[d], "qty": qty, "new_hours": [],
                                            "_hour_out": ho}, 0.01 * abs(d - s)))
            if not evaluate(cands):
                break

    # ---- 2단계: 다른 날로 하루 생산량 일부 이동 ----
    def day_transfer(src_day, dst_day):
        on, prod = cur["on"].to_numpy(), cur["production"].to_numpy()
        si, di = day_idx[src_day], day_idx[dst_day]
        qty = min(DAY_STEP * day_total0[src_day], max_move_share * day_total0[src_day] - day_out[src_day])
        if qty < 1:
            return None
        # 보내는 날: 예측 상위 시간부터, 시간마다 원래 생산량의 max_move_share까지
        take = np.zeros(len(t))
        need = qty
        for i in sorted([i for i in si if on[i] == 1], key=lambda i: -pred[i]):
            room = min(prod[i], max_move_share * orig_prod[i] - hour_out[i] - take[i])
            if room <= 0:
                continue
            x = min(room, need)
            take[i] += x
            need -= x
            if need <= 1e-9:
                break
        if need > 1e-6:
            return None
        # 받는 날: 예측 낮은 가동 시간부터 설비 한도까지, 모자라면 (허용 시) 가동 구간을 선례 있는 시간으로 연장
        p = cur.copy()
        newon = p["on"].to_numpy().copy()
        newprod = prod - take
        left = qty
        slots = sorted([i for i in di if newon[i] == 1], key=lambda i: pred[i])
        new_hours = []
        while left > 1e-9:
            placed = False
            for i in slots:
                if pred[i] + ctx.margin >= ctx.cap - DEST_HEADROOM:
                    continue
                room = ctx.prod_cap - newprod[i]
                if room <= 0:
                    continue
                x = min(room, left)
                newprod[i] += x
                left -= x
                placed = True
                if left <= 1e-9:
                    break
            if left <= 1e-9:
                break
            if not allow_new:
                return None
            ext = [i for i in di if newon[i] == 0 and precedent[i]
                   and any(newon[j] == 1 for j in (i - 1, i + 1) if j in di)]
            if not ext:
                return None
            j = min(ext, key=lambda i: i)            # 앞에서부터 하나씩 연장
            newon[j] = 1
            new_hours.append(t[j])
            slots = [j]
            if not placed and len(new_hours) > 12:
                return None
        p["on"] = newon
        p["production"] = newprod
        info = {"kind": "day", "from": src_day, "to": dst_day, "qty": qty, "new_hours": new_hours,
                "_hour_out": take, "_from_day": src_day}
        pen = CROSS_DAY_PENALTY + NEW_HOUR_PENALTY * len(new_hours)
        return (p, [src_day, dst_day], info, pen)

    if not same_day_only:
        for _ in range(MAX_ITER):
            day_max = pd.Series(pred, index=t).groupby(day).max()
            order = list(day_max.sort_values(ascending=False).index)
            cands = []
            for src in order[:3]:
                for dst in order[3:]:
                    if day_max[dst] >= day_max[src] - 5:
                        continue
                    c = day_transfer(src, dst)
                    if c is not None:
                        cands.append(c)
            if not evaluate(cands):
                break
    return cur, pred, moves


def merge_moves(moves: list[dict]) -> pd.DataFrame:
    """같은 출발·도착끼리 합친 이동 목록. kind = hour(같은 날 시간 이동) / day(다른 날로 하루 생산량 이동)."""
    cols = ["kind", "from", "to", "qty", "new_hours"]
    if not moves:
        return pd.DataFrame(columns=cols)
    m = pd.DataFrame(moves)[cols]
    return (m.groupby(["kind", "from", "to"], as_index=False, sort=False)
             .agg(qty=("qty", "sum"), new_hours=("new_hours", lambda s: sorted({h for x in s for h in x})))
             .sort_values(["kind", "from", "to"], ascending=[False, True, True]).reset_index(drop=True))


# =============================================================================
# 출력
# =============================================================================
def summarize(ctx, plan, revised, pred0, pred1, moves, price):
    t = plan.index
    thr = ctx.peak_thr
    day = t.normalize()
    actual = ctx.df["power"].reindex(t).to_numpy() if ctx.backtest else None
    act_after = np.round(actual + (pred1 - pred0), 1) if actual is not None else None
    mm = merge_moves(moves)
    by_day = []
    for d0 in sorted(set(day)):
        m = day == d0
        row = {"date": f"{d0:%Y-%m-%d}", "dow": int(d0.dayofweek),
               "on_before": int(plan["on"][m].sum()), "on_after": int(revised["on"][m].sum()),
               "prod_before": float(plan["production"][m].sum()), "prod_after": float(revised["production"][m].sum()),
               "pred_max_before": round(float(pred0[m].max()), 1), "pred_max_after": round(float(pred1[m].max()), 1),
               "risk_before": int((pred0[m] + ctx.margin >= thr).sum()), "risk_after": int((pred1[m] + ctx.margin >= thr).sum())}
        if actual is not None:
            row.update({"actual_max": float(np.nanmax(actual[m])), "actual_max_after_est": float(np.nanmax(act_after[m])),
                        "actual_peak_hours": int((actual[m] >= thr).sum()),
                        "actual_peak_hours_after_est": int((act_after[m] >= thr).sum())})
        by_day.append(row)
    week_cut = float(pred0.max() - pred1.max())
    s = {"start": f"{t.min():%Y-%m-%d}", "end": f"{t.max():%Y-%m-%d}", "backtest": ctx.backtest, "model": ctx.model_file,
         "peak_threshold": thr, "margin": ctx.margin,
         "pred_week_max_before": round(float(pred0.max()), 1), "pred_week_max_after": round(float(pred1.max()), 1),
         "days_lowered": int(sum(r["pred_max_after"] < r["pred_max_before"] - 0.05 for r in by_day)),
         "avg_cut_on_lowered_days": round(float(np.mean([r["pred_max_before"] - r["pred_max_after"] for r in by_day
                                                          if r["pred_max_after"] < r["pred_max_before"] - 0.05] or [0])), 1),
         "max_rise_on_receiving_days": round(float(max([r["pred_max_after"] - r["pred_max_before"] for r in by_day] + [0])), 1),
         "risk_hours_before": int((pred0 + ctx.margin >= thr).sum()), "risk_hours_after": int((pred1 + ctx.margin >= thr).sum()),
         "moved_production": round(float(mm["qty"].sum()), 0), "total_production": round(float(plan["production"].sum()), 0),
         "n_moves": int(len(mm)), "new_hours": int((revised["on"] - plan["on"]).clip(lower=0).sum()),
         "price": price, "monthly_saving_est": round(max(week_cut, 0) * price, 0)}
    if actual is not None:
        s.update({"actual_week_max": float(np.nanmax(actual)), "actual_week_max_after_est": float(np.nanmax(act_after)),
                  "actual_peak_hours": int((actual >= thr).sum()), "actual_peak_hours_after_est": int((act_after >= thr).sum())})
    return s, by_day, mm, actual, act_after


def write_outputs(ctx, plan, revised, pred0, pred1, moves, out_dir, filled, allow_new, price):
    os.makedirs(out_dir, exist_ok=True)
    t = plan.index
    thr = ctx.peak_thr
    s, by_day, mm, actual, act_after = summarize(ctx, plan, revised, pred0, pred1, moves, price)

    hourly = pd.DataFrame({
        "pred_before": np.round(pred0, 1), "pred_after": np.round(pred1, 1),
        "on_before": plan["on"].to_numpy(), "production_before": plan["production"].round(0).to_numpy(),
    })
    if actual is not None:
        hourly["actual"] = actual
        hourly["actual_after_est"] = act_after
    pt.write_plan_csv(revised, os.path.join(out_dir, "revised_plan.csv"))
    pt.write_plan_csv(revised, os.path.join(out_dir, "advice_hourly.csv"), extra=hourly)
    mm.assign(**{"from": mm["from"].astype(str), "to": mm["to"].astype(str), "qty": mm["qty"].round(0),
                 "new_hours": mm["new_hours"].map(lambda x: " ".join(f"{h:%m-%d %H}시" for h in x))}).to_csv(
        os.path.join(out_dir, "moves.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": s, "days": by_day, "allow_new_hours": allow_new}, f, ensure_ascii=False, indent=1)

    L = [f"# 주간 계획 수정 제안: {s['start']} ~ {s['end']}", ""]
    L.append(f"- 모델: `{ctx.model_file}` (1주 전 예측, 가동·생산 계획 연계)")
    L.append(f"- 목표: 날마다의 예측 최대전력 낮추기. 위험 시간 = 예측 + 안전 마진 {ctx.margin:.1f} kW ≥ {thr:g} kW")
    L.append(f"- 수단: 생산량 이동{' + 가동 시간대 연장(실험적)' if allow_new else ''}. 주간 총 생산량은 그대로입니다.")
    if filled:
        L.append(f"- 생산량이 비어 있던 {filled}시간은 최근 4주 같은 요일·시각 평균으로 채웠습니다.")
    if ctx.backtest:
        L.append("- 백테스트: 계획 시작 전까지의 전력만 써서 예측했습니다. '수정 후 실제(추정)' = 실제 전력 + 모델이 예측한 변화량.")
    L += ["", "## 요약", ""]
    L.append(f"- 주간 예측 최대: {s['pred_week_max_before']} → {s['pred_week_max_after']} kW")
    L.append(f"- 일 예측 최대가 내려간 날: {s['days_lowered']}일, 평균 −{s['avg_cut_on_lowered_days']} kW "
             f"(생산량을 받은 날은 최대 +{s['max_rise_on_receiving_days']} kW)")
    L.append(f"- 위험 시간: {s['risk_hours_before']} → {s['risk_hours_after']}시간")
    L.append(f"- 옮긴 생산량: {s['moved_production']:.0f} (주간 총 {s['total_production']:.0f}의 "
             f"{100 * s['moved_production'] / max(s['total_production'], 1):.1f}%), 이동 {s['n_moves']}건"
             + (f", 새로 켠 가동 시간 {s['new_hours']}시간" if s['new_hours'] else ""))
    if actual is not None:
        L.append(f"- 실제 주간 최대: {s['actual_week_max']:.0f} → 수정 후 추정 {s['actual_week_max_after_est']:.0f} kW, "
                 f"실제 피크 시간(≥{thr:g}): {s['actual_peak_hours']} → 추정 {s['actual_peak_hours_after_est']}시간")
    L.append(f"- 기본요금 (예시 단가 {price:,}원/kW·월): 주간 예측 최대가 그대로 요금 기준이 된다고 보면 월 {s['monthly_saving_est']:,.0f}원")
    L += ["", "## 날짜별", ""]
    head = "| 날짜 | 가동 시간 | 생산량 (전→후) | 예측 최대 (전→후) | 위험 시간 (전→후) |"
    sep = "|---|---|---|---|---|"
    if actual is not None:
        head += " 실제 최대 → 수정 후 추정 |"
        sep += "---|"
    L += [head, sep]
    for r in by_day:
        d0 = pd.Timestamp(r["date"])
        row = (f"| {d0:%m/%d}({pt.DOW_KO[r['dow']]}) | {r['on_before']} → {r['on_after']} | "
               f"{r['prod_before']:.0f} → {r['prod_after']:.0f} | {r['pred_max_before']} → {r['pred_max_after']} | "
               f"{r['risk_before']} → {r['risk_after']} |")
        if actual is not None:
            row += f" {r['actual_max']:.0f} → {r['actual_max_after_est']:.0f} |"
        L.append(row)
    L += ["", "## 수정 내용", ""]
    if mm.empty:
        L.append("- 규칙 안에서 최대전력을 의미 있게 낮추는 이동을 찾지 못했습니다.")
    day_of = t.normalize()
    for _, m in mm.iterrows():
        if m["kind"] == "day":
            fd, td = m["from"], m["to"]
            fm, tm = day_of == fd, day_of == td
            taken = (plan["production"] - revised["production"]).clip(lower=0)[fm]
            hours = [h.hour for h, v in taken.items() if v >= 1]
            line = (f"- {fd:%m/%d}({pt.DOW_KO[fd.dayofweek]}) → {td:%m/%d}({pt.DOW_KO[td.dayofweek]}): 생산량 {m['qty']:.0f} 이동 "
                    f"(하루 생산량의 {100 * m['qty'] / max(plan['production'][fm].sum(), 1):.0f}%, "
                    f"{fd:%m/%d} {min(hours) if hours else 0}~{max(hours) if hours else 0}시에서 덜어냄)  "
                    f"[일 예측 최대 {fd:%m/%d} {pred0[fm].max():.1f}→{pred1[fm].max():.1f}, "
                    f"{td:%m/%d} {pred0[tm].max():.1f}→{pred1[tm].max():.1f}]")
            if m["new_hours"]:
                line += (f"\n  - 받는 날 가동 연장 (실험적): {td:%m/%d} " + ", ".join(f"{h:%H}시" for h in m["new_hours"])
                         + " 를 새로 가동")
        else:
            a_, b_ = t.get_loc(m["from"]), t.get_loc(m["to"])
            line = (f"- {fmt_t(m['from'])} → {fmt_t(m['to'])}: 생산량 {m['qty']:.0f} 이동 (같은 날)  "
                    f"[예측 {fmt_t(m['from'])} {pred0[a_]:.1f}→{pred1[a_]:.1f}, {fmt_t(m['to'])} {pred0[b_]:.1f}→{pred1[b_]:.1f}]")
        L.append(line)
    L += ["", "## 읽을 때 주의", ""]
    L.append("- 이 데이터에서 평일 낮 전력은 생산량보다 가동 자체에 크게 좌우됩니다. 생산량을 옮겨 줄일 수 있는 폭은 몇 kW 수준이고, "
             "남는 피크는 당일 M2 피크 경보로 대응해야 합니다.")
    L.append("- 모델은 시간별 생산량보다 그날 총 생산량에 더 반응합니다. 그래서 다른 날로 옮기는 안이 주로 효과를 냅니다.")
    if allow_new:
        L.append("- 가동 시간대 연장은 과거에 그 요일·시각에 가동한 선례(5% 이상)가 있는 시간만 허용했지만, 선례가 적어 예측 신뢰도가 낮습니다.")
    L.append("- 공정 순서, 납기, 인력·인건비(야간 1.5배) 제약은 데이터에 없어 반영하지 않았습니다. 현장에서 가능한 이동인지 확인하세요.")
    text = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "advice.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


# =============================================================================
def make_template(start: str, days: int, source: str, folder: str | None = None) -> str:
    start = pd.Timestamp(start)
    idx = pd.date_range(start, periods=24 * days, freq="h")
    df, copied, hour_thr, _ = pt.load_history()
    executed = pt.executed_plan(df, copied, hour_thr)
    if source == "actual":
        if not idx.isin(df.index).all():
            raise ValueError("--source actual 은 데이터에 있는 날짜만 쓸 수 있습니다.")
        plan = executed.loc[idx]
    else:
        plan = pt.usual_plan(df, executed, copied, start, idx)
    folder = folder or PLANS_DIR
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"plan_{start:%Y-%m-%d}{'_actual' if source == 'actual' else ''}.csv")
    pt.write_plan_csv(plan, path)
    print(f"[템플릿] {os.path.relpath(path, HERE)} ({days}일, "
          f"{'실제 실행 계획' if source == 'actual' else '요일별 평소 일정 + 최근 4주 평균 생산량'})")
    print("         on(가동 1/0)과 production(생산량)을 고친 뒤 --plan 으로 넣으세요. dow 열은 참고용입니다.")
    return path


def advise(path: str, same_day_only=False, allow_new=False, max_move_share=0.5, window=6, out_dir=None,
           model=None, price=DEFAULT_PRICE, quiet=False):
    plan = pt.read_plan_csv(path)
    ctx = Context(plan.index.min(), model)
    check_window(ctx, plan)
    plan, filled = fill_production(ctx, plan)
    base = pt.week_features(ctx.history["power"], ctx.history, plan.index)
    assert base.notna().all().all(), "1주 전 feature에 빈 값이 있습니다."
    pred0 = predict_many(ctx, base, [plan])[0]
    part, rows = predict_days(ctx, base, [plan], sorted(set(plan.index.normalize()))[1:3])
    assert np.allclose(part[0], pred0[rows]), "날 단위 부분 예측이 전체 예측과 다릅니다."
    revised, pred1, moves = search(ctx, plan, base, same_day_only, allow_new, max_move_share, window)
    assert np.allclose(predict_many(ctx, base, [revised])[0], pred1), "탐색 중 예측과 최종 예측이 다릅니다."
    assert abs(revised["production"].sum() - plan["production"].sum()) < 1e-6, "주간 총 생산량이 바뀌었습니다."
    name = os.path.splitext(os.path.basename(path))[0]
    out_dir = out_dir or os.path.join(PLANS_DIR, "out", name)
    text = write_outputs(ctx, plan, revised, pred0, pred1, moves, out_dir, filled, allow_new, price)
    if not quiet:
        print(text)
        print(f"[저장] {os.path.relpath(out_dir, HERE)}/ revised_plan.csv, advice_hourly.csv, moves.csv, summary.json, advice.md")
    return out_dir


# 웹 화면(12_build_report.py)에 넣을 예시: 백테스트 1주 + 다음 1주, 기본안·적극안
WEB_DIR = os.path.join(PLANS_DIR, "out", "web")
WEB_EXAMPLES = [
    # key, 시작일, 템플릿 출처, 적극안 여부
    ("backtest_0908", "2021-09-08", "actual", False),
    ("backtest_0908_ext", "2021-09-08", "actual", True),
    ("week_0915", "2021-09-15", "usual", False),
    ("week_0915_ext", "2021-09-15", "usual", True),
]
MODES = {False: {"allow_new": False, "max_move_share": 0.5},   # 기본안: 생산량 이동, 시간·하루당 최대 50%
         True: {"allow_new": True, "max_move_share": 0.8}}     # 적극안: 최대 80% + 가동 시간대 연장(실험적)


def web_examples():
    for key, start, source, ext in WEB_EXAMPLES:
        path = make_template(start, 7, source, folder=os.path.join(WEB_DIR, "input"))  # 사용자가 고친 plans/*.csv는 건드리지 않음
        out = advise(path, allow_new=MODES[ext]["allow_new"], max_move_share=MODES[ext]["max_move_share"],
                     out_dir=os.path.join(WEB_DIR, key), quiet=True)
        with open(os.path.join(out, "summary.json"), encoding="utf-8") as f:
            sm = json.load(f)["summary"]
        print(f"[웹 예시] {key}: 주간 예측 최대 {sm['pred_week_max_before']} -> {sm['pred_week_max_after']} kW, "
              f"내려간 날 {sm['days_lowered']}일 평균 -{sm['avg_cut_on_lowered_days']} kW, 이동 {sm['n_moves']}건")
    with open(os.path.join(WEB_DIR, "modes.json"), "w", encoding="utf-8") as f:
        json.dump({"examples": [{"key": k, "start": s, "source": src, "ext": e} for k, s, src, e in WEB_EXAMPLES],
                   "modes": {"base": MODES[False], "ext": MODES[True]}}, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser(description="주간 계획을 넣으면 일 최대전력을 낮추는 수정안을 냅니다.")
    ap.add_argument("--template", help="이 날짜(YYYY-MM-DD)부터 계획 템플릿 생성")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--source", choices=["usual", "actual"], default="usual")
    ap.add_argument("--plan", help="계획 CSV 경로")
    ap.add_argument("--same-day-only", action="store_true", help="같은 날 안에서만 옮김")
    ap.add_argument("--allow-new-hours", action="store_true", help="선례가 있는 비가동 시간을 켜서 가동 구간 연장 (실험적)")
    ap.add_argument("--max-move-share", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=6, help="같은 날 안에서 옮길 수 있는 거리(시간)")
    ap.add_argument("--price", type=int, default=DEFAULT_PRICE, help="기본요금 예시 단가 (원/kW·월)")
    ap.add_argument("--out", help="결과 폴더 (기본 plans/out/<계획 파일 이름>)")
    ap.add_argument("--web-examples", action="store_true", help="웹 화면용 예시 4개 생성 -> plans/out/web/ (12가 읽음)")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if a.template:
        make_template(a.template, a.days, a.source)
    if a.plan:
        advise(a.plan, a.same_day_only, a.allow_new_hours, a.max_move_share, a.window, a.out, price=a.price)
    if a.web_examples:
        web_examples()
    if not a.template and not a.plan and not a.web_examples:
        ap.print_help()


if __name__ == "__main__":
    main()
