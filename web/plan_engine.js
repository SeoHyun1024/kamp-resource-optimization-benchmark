/*
 * 주간 계획 엔진 (브라우저·Node 공용): 14_plan_advisor.py 의 예측·수정안 탐색을 그대로 옮긴 것.
 *   - 데이터: web/data/plan_editor.js (14_plan_advisor.py --web-editor 가 생성)
 *       LightGBM 트리 배열, 편집할 주의 1주 전 feature(과거 전력으로 이미 정해진 값), 템플릿 계획, 탐색 상수
 *   - 계획 feature 계산(plan_tools.plan_features), 점수(score), 탐욕 탐색(search), 요약(summarize)을 파이썬과 같은 순서로 계산한다.
 *   - 파이썬 결과와 같은지는 저장소 맨 위의 check_plan_engine.mjs 로 확인한다 (node check_plan_engine.mjs).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.PlanEngine = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ── 트리 예측 ────────────────────────────────────────────
  // 트리: [split_feature[], threshold[], left[], right[], leaf_value[], root] (자식이 음수면 ~잎 번호)
  // 가능하면 트리를 if 코드로 바꿔 한 번 컴파일한다 (배열을 따라가는 것보다 몇 배 빠름). 실패하면 배열로 계산.
  function compileModel(trees) {
    const evalArrays = (x) => {
      let s = 0;
      for (let k = 0; k < trees.length; k++) {
        const [F, T, L, R, V, root] = trees[k];
        let n = root;
        while (n >= 0) n = x[F[n]] <= T[n] ? L[n] : R[n];
        s += V[~n];
      }
      return s;
    };
    try {
      const num = (v) => (Object.is(v, -0) ? '-0' : String(v));   // String(double)은 왕복 정확
      const node = (t, n) => {
        const [F, T, L, R, V] = t;
        if (n < 0) return num(V[~n]);
        return `(x[${F[n]}]<=${num(T[n])}?${node(t, L[n])}:${node(t, R[n])})`;
      };
      // 트리 순서대로 하나씩 더한다 (LightGBM·파이썬과 같은 덧셈 순서 -> 같은 값)
      const body = ['let s=0;'].concat(trees.map((t) => `s+=${node(t, t[5])};`), ['return s;']).join('\n');
      const fn = new Function('x', body);
      return fn;
    } catch (e) {
      return evalArrays;
    }
  }

  // ── 계획 feature (plan_tools.plan_features 와 같음) ───────────
  const PLAN_COLS = ['plan_on', 'plan_on_prev', 'plan_on_next', 'plan_hours_since_on', 'plan_hours_until_off', 'plan_on_hours',
    'plan_first_on_hour', 'plan_last_on_hour', 'plan_day_type', 'plan_prod', 'plan_prod_roll3', 'plan_prod_day'];
  const DAY_TYPES = ['비가동', '종일 가동', '가동 시작일', '가동 종료일', '중간 정지', '부분 가동'];

  function planFeatures(on, production) {
    const n = on.length, nd = n / 24;
    const f = {};
    PLAN_COLS.forEach((c) => { f[c] = new Float64Array(n); });
    const prod = new Float64Array(n);
    for (let i = 0; i < n; i++) prod[i] = on[i] === 1 ? +production[i] : 0;
    for (let d = 0; d < nd; d++) {
      const a = d * 24, b = a + 24;
      let onHours = 0, first = -1, last = -1, pday = 0;
      for (let i = a; i < b; i++) {
        onHours += on[i];
        if (on[i] === 1) { if (first < 0) first = i - a; last = i - a; }
        pday += prod[i];
      }
      // 같은 날 안에서 꺼질 때마다 새 구간: 구간 안 누적 가동 시간 (앞에서 / 뒤에서)
      let run = 0, cum = 0, prevRun = -1;
      for (let i = a; i < b; i++) {
        if (on[i] === 0) run++;
        if (run !== prevRun) { cum = 0; prevRun = run; }
        cum += on[i];
        f.plan_hours_since_on[i] = cum;
      }
      run = 0; cum = 0; prevRun = -1;
      for (let i = b - 1; i >= a; i--) {
        if (on[i] === 0) run++;
        if (run !== prevRun) { cum = 0; prevRun = run; }
        cum += on[i];
        f.plan_hours_until_off[i] = cum;
      }
      const start = on[a] === 0 && onHours > 0, end = on[b - 1] === 0 && onHours > 0;
      const type = onHours === 0 ? '비가동' : start && end ? '부분 가동' : start ? '가동 시작일' : end ? '가동 종료일' : onHours < 24 ? '중간 정지' : '종일 가동';
      for (let i = a; i < b; i++) {
        f.plan_on_hours[i] = onHours;
        f.plan_first_on_hour[i] = first;
        f.plan_last_on_hour[i] = last;
        f.plan_day_type[i] = DAY_TYPES.indexOf(type);
        f.plan_prod_day[i] = pday;
      }
    }
    for (let i = 0; i < n; i++) {
      f.plan_on[i] = on[i];
      f.plan_on_prev[i] = i > 0 ? on[i - 1] : on[i];
      f.plan_on_next[i] = i < n - 1 ? on[i + 1] : on[i];
      f.plan_prod[i] = prod[i];
      // rolling(3, center=True, min_periods=1).mean()
      let s = prod[i], c = 1;
      if (i > 0) { s = prod[i - 1] + s; c++; }
      if (i < n - 1) { s += prod[i + 1]; c++; }
      f.plan_prod_roll3[i] = s / c;
    }
    return f;
  }

  // ── 엔진 ────────────────────────────────────────────────
  function create(data) {
    const models = {};
    for (const k of Object.keys(data.models)) models[k] = compileModel(data.models[k]);
    const C = data.consts;

    function weekCtx(key) {
      const w = data.weeks[key];
      if (!w) throw new Error('편집할 수 없는 주: ' + key);
      if (!w._x) {
        // feature 벡터 틀: base 값은 고정, plan 값만 행마다 채운다
        const idxOf = {};
        data.features.forEach((c, j) => { idxOf[c] = j; });
        w._baseIdx = w.base_features.map((c) => idxOf[c]);
        w._planIdx = PLAN_COLS.map((c) => idxOf[c]);
        w._x = w.base.map((row) => {
          const x = new Float64Array(data.features.length);
          row.forEach((v, j) => { x[w._baseIdx[j]] = v; });
          return x;
        });
        w._predict = models[w.model];
      }
      return w;
    }

    // rows 만 다시 예측 (나머지는 그대로)
    function predictRows(w, on, prod, rows) {
      const f = planFeatures(on, prod), out = new Float64Array(rows.length);
      for (let r = 0; r < rows.length; r++) {
        const i = rows[r], x = w._x[i];
        for (let j = 0; j < PLAN_COLS.length; j++) x[w._planIdx[j]] = f[PLAN_COLS[j]][i];
        out[r] = w._predict(x);
      }
      return out;
    }
    function predict(key, on, prod) {
      const w = weekCtx(key), rows = Array.from({ length: on.length }, (_, i) => i);
      return predictRows(w, on, prod, rows);
    }

    function score(w, pred) {
      const n = pred.length, TAU = C.TAU;
      let smax = 0;
      for (let a = 0; a < n; a += 24) {
        let m = -Infinity;
        for (let i = a; i < a + 24; i++) if (pred[i] > m) m = pred[i];
        let s = 0;
        for (let i = a; i < a + 24; i++) s += Math.exp((pred[i] - m) / TAU);
        smax += m + TAU * Math.log(s);
      }
      let mw = -Infinity;
      for (let i = 0; i < n; i++) if (pred[i] > mw) mw = pred[i];
      let sw = 0, exc = 0;
      for (let i = 0; i < n; i++) {
        sw += Math.exp((pred[i] - mw) / TAU);
        exc += Math.max(0, pred[i] + w.margin - w.cap);
      }
      return smax + C.WEEK_WEIGHT * (mw + TAU * Math.log(sw)) + C.EXCESS_WEIGHT * exc;
    }

    // 14_plan_advisor.search 와 같은 순서·조건
    // 비동기: opt.yield 가 있으면 후보 몇 개마다 화면에 양보한다(브라우저가 멈춘 것처럼 보이지 않게). 결과는 같다.
    async function search(key, on0, prod0, opt) {
      const pause = opt.yield || null, progress = opt.onProgress || null;
      let evals = 0;
      const w = weekCtx(key);
      const allowNew = !!opt.allowNew, mms = opt.maxMoveShare, window = opt.window ?? C.WINDOW, sameDayOnly = !!opt.sameDayOnly;
      const n = on0.length, nd = n / 24;
      let on = Int32Array.from(on0), prod = Float64Array.from(prod0.map(Number));
      for (let i = 0; i < n; i++) if (on[i] === 0) prod[i] = 0;
      const orig = Float64Array.from(prod);
      let pred = predict(key, on, prod);
      const pred0 = Float64Array.from(pred);
      const moves = [];
      const dayIdx = Array.from({ length: nd }, (_, d) => Array.from({ length: 24 }, (_, h) => d * 24 + h));
      const dayTotal0 = dayIdx.map((ix) => ix.reduce((s, i) => s + orig[i], 0));
      const dayOut = new Float64Array(nd);
      const hourOut = new Float64Array(n);
      const precedent = w.precedent;
      const dayOf = (i) => Math.floor(i / 24);
      let curScore = score(w, pred);

      async function evaluate(cands) {
        if (!cands.length) return false;
        const days = [...new Set(cands.flatMap((c) => c.days))].sort((a, b) => a - b);
        const rows = days.flatMap((d) => dayIdx[d]);
        let best = -1, bestGain = -Infinity, bestPred = null;
        for (let k = 0; k < cands.length; k++) {
          const c = cands[k];
          const p = predictRows(w, c.on, c.prod, rows);
          const full = Float64Array.from(pred);
          for (let r = 0; r < rows.length; r++) full[rows[r]] = p[r];
          const g = curScore - score(w, full) - c.pen;
          if (pause && ++evals % 15 === 0) { if (progress) progress({ evals, moves: moves.length }); await pause(); }
          if (g > bestGain) { bestGain = g; best = k; bestPred = full; }   // 같은 값이면 앞 후보 (np.argmax와 같음)
        }
        if (bestGain <= C.MIN_GAIN) return false;
        const c = cands[best];
        on = c.on; prod = c.prod; pred = bestPred; curScore = score(w, pred);
        for (let i = 0; i < n; i++) hourOut[i] += c.hourOut[i];
        if (c.info.kind === 'day') dayOut[c.fromDay] += c.info.qty;
        moves.push(c.info);
        return true;
      }

      // 1단계: 같은 날
      for (let d0 = 0; d0 < nd; d0++) {
        const idx = dayIdx[d0];
        for (let it = 0; it < C.MAX_ITER; it++) {
          const srcs = idx.filter((i) => on[i] === 1 && prod[i] >= 1 && hourOut[i] < mms * orig[i] - 1e-9)
            .sort((a, b) => pred[b] - pred[a]).slice(0, 6);
          const cands = [];
          for (const s of srcs) {
            const qty = Math.min(C.STEP_SHARE * orig[s], mms * orig[s] - hourOut[s], prod[s]);
            for (const d of idx) {
              if (qty < 1 || d === s || on[d] === 0 || Math.abs(d - s) > window || pred[d] >= pred[s] - 5
                || pred[d] + w.margin >= w.cap - C.DEST_HEADROOM || prod[d] + qty > w.prod_cap) continue;
              const p = Float64Array.from(prod);
              p[s] -= qty; p[d] += qty;
              const ho = new Float64Array(n); ho[s] = qty;
              cands.push({ on, prod: p, days: [d0], pen: 0.01 * Math.abs(d - s), hourOut: ho,
                info: { kind: 'hour', from: w.time[s], to: w.time[d], qty, new_hours: [] } });
            }
          }
          if (!(await evaluate(cands))) break;
        }
      }

      // 2단계: 다른 날
      function dayTransfer(src, dst, mult) {
        const si = dayIdx[src], di = dayIdx[dst];
        const qty = Math.min(mult * C.DAY_STEP * dayTotal0[src], mms * dayTotal0[src] - dayOut[src]);
        if (qty < 1) return null;
        const take = new Float64Array(n);
        let need = qty;
        const order = si.filter((i) => on[i] === 1).sort((a, b) => pred[b] - pred[a]);
        for (const i of order) {
          const room = Math.min(prod[i], mms * orig[i] - hourOut[i] - take[i]);
          if (room <= 0) continue;
          const x = Math.min(room, need);
          take[i] += x; need -= x;
          if (need <= 1e-9) break;
        }
        if (need > 1e-6) return null;
        const newon = Int32Array.from(on);
        const newprod = new Float64Array(n);
        for (let i = 0; i < n; i++) newprod[i] = prod[i] - take[i];
        let left = qty;
        let slots = di.filter((i) => newon[i] === 1).sort((a, b) => pred[a] - pred[b]);
        const newHours = [];
        while (left > 1e-9) {
          let placed = false;
          for (const i of slots) {
            if (pred[i] + w.margin >= w.cap - C.DEST_HEADROOM) continue;
            const room = w.prod_cap - newprod[i];
            if (room <= 0) continue;
            const x = Math.min(room, left);
            newprod[i] += x; left -= x; placed = true;
            if (left <= 1e-9) break;
          }
          if (left <= 1e-9) break;
          if (!allowNew) return null;
          const inDay = (j) => j >= di[0] && j <= di[di.length - 1];
          const ext = di.filter((i) => newon[i] === 0 && precedent[i] === 1 && [i - 1, i + 1].some((j) => inDay(j) && newon[j] === 1));
          if (!ext.length) return null;
          const j = Math.min(...ext);
          newon[j] = 1;
          newHours.push(w.time[j]);
          slots = [j];
          if (!placed && newHours.length > 12) return null;
        }
        return { on: newon, prod: newprod, days: [src, dst], hourOut: take, fromDay: src,
          pen: C.CROSS_DAY_PENALTY + C.NEW_HOUR_PENALTY * newHours.length + 0.3 * (mult - 1),
          info: { kind: 'day', from: w.time[si[0]], to: w.time[di[0]], qty, new_hours: newHours } };
      }
      if (!sameDayOnly) {
        for (let it = 0; it < C.MAX_ITER; it++) {
          const dayMax = dayIdx.map((ix) => Math.max(...ix.map((i) => pred[i])));
          const order = Array.from({ length: nd }, (_, d) => d).sort((a, b) => dayMax[b] - dayMax[a]);
          const cands = [];
          for (const src of order.slice(0, 3)) {
            for (const dst of order.slice(3)) {
              if (dayMax[dst] >= dayMax[src] - 5) continue;
              for (const mult of [1, 2, 3]) {
                const c = dayTransfer(src, dst, mult);
                if (c) cands.push(c);
              }
            }
          }
          if (!(await evaluate(cands))) break;
        }
      }
      return { on: Array.from(on), production: Array.from(prod), pred0: Array.from(pred0), pred1: Array.from(pred), moves };
    }

    // 웹 화면용 결과 (12_build_report 가 만드는 주간 계획 결과와 같은 모양)
    function summarize(key, planOn, planProd, res, opt) {
      const w = weekCtx(key), thr = w.peak_threshold, n = planOn.length, nd = n / 24;
      const p0 = res.pred0, p1 = res.pred1;
      const prodB = planOn.map((o, i) => (o === 1 ? +planProd[i] : 0));
      const act = w.actual, actAfter = act ? act.map((v, i) => Math.round((v + (p1[i] - p0[i])) * 10) / 10) : null;
      const r1 = (v) => Math.round(v * 10) / 10;
      const days = [];
      for (let d = 0; d < nd; d++) {
        const ix = Array.from({ length: 24 }, (_, h) => d * 24 + h);
        const mx = (arr) => Math.max(...ix.map((i) => arr[i]));
        const date = w.time[d * 24].slice(0, 10);
        const row = {
          date, dow: (new Date(date + 'T00:00:00Z').getUTCDay() + 6) % 7,
          on_before: ix.reduce((s, i) => s + planOn[i], 0), on_after: ix.reduce((s, i) => s + res.on[i], 0),
          prod_before: ix.reduce((s, i) => s + prodB[i], 0), prod_after: ix.reduce((s, i) => s + res.production[i], 0),
          pred_max_before: r1(mx(p0)), pred_max_after: r1(mx(p1)),
          risk_before: ix.filter((i) => p0[i] + w.margin >= thr).length, risk_after: ix.filter((i) => p1[i] + w.margin >= thr).length,
        };
        row.out_of_range = row.prod_before > w.day_prod_p99;
        if (act) {
          Object.assign(row, { actual_max: mx(act), actual_max_after_est: mx(actAfter),
            actual_peak_hours: ix.filter((i) => act[i] >= thr).length, actual_peak_hours_after_est: ix.filter((i) => actAfter[i] >= thr).length });
        }
        days.push(row);
      }
      // 같은 출발·도착끼리 합치기 (14 merge_moves)
      const merged = new Map();
      for (const m of res.moves) {
        const k = `${m.kind}|${m.from}|${m.to}`;
        const e = merged.get(k) || { kind: m.kind, from: m.from, to: m.to, qty: 0, new_hours: [] };
        e.qty += m.qty;
        e.new_hours = [...new Set(e.new_hours.concat(m.new_hours))].sort();
        merged.set(k, e);
      }
      const moves = [...merged.values()];
      const lowered = days.filter((r) => r.pred_max_after < r.pred_max_before - 0.05);
      const moved = moves.reduce((s, m) => s + m.qty, 0), total = prodB.reduce((s, v) => s + v, 0);
      const maxB = Math.max(...p0), maxA = Math.max(...p1);
      const summary = {
        start: w.start, end: w.end, backtest: w.backtest, model: w.model, peak_threshold: thr, margin: w.margin,
        pred_week_max_before: r1(maxB), pred_week_max_after: r1(maxA),
        days_lowered: lowered.length,
        avg_cut_on_lowered_days: r1(lowered.length ? lowered.reduce((s, r) => s + r.pred_max_before - r.pred_max_after, 0) / lowered.length : 0),
        max_rise_on_receiving_days: r1(Math.max(0, ...days.map((r) => r.pred_max_after - r.pred_max_before))),
        risk_hours_before: p0.filter((v) => v + w.margin >= thr).length, risk_hours_after: p1.filter((v) => v + w.margin >= thr).length,
        moved_production: Math.round(moved), total_production: Math.round(total), n_moves: moves.length,
        new_hours: res.on.reduce((s, o, i) => s + Math.max(0, o - planOn[i]), 0),
        day_prod_p99: Math.round(w.day_prod_p99), out_of_range_days: days.filter((r) => r.out_of_range).map((r) => r.date),
      };
      if (act) Object.assign(summary, { actual_week_max: Math.max(...act), actual_week_max_after_est: Math.max(...actAfter),
        actual_peak_hours: act.filter((v) => v >= thr).length, actual_peak_hours_after_est: actAfter.filter((v) => v >= thr).length });
      return {
        start: w.start, ext: !!(opt && opt.allowNew), summary, days, time: w.time, moves,
        hours: { pred_before: p0.map(r1), pred_after: p1.map(r1), production_before: prodB.map(Math.round), production: res.production.map(Math.round),
          actual: act, actual_after_est: actAfter, on_before: Array.from(planOn), on_after: Array.from(res.on) },
      };
    }

    return { data, predict, search, summarize, planFeatures, weekCtx };
  }

  return { create, planFeatures, PLAN_COLS };
});
