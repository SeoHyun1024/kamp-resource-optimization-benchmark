/* 자원 최적화 에이전트 화면. 서버 없이 data/agent_data.js(12_build_report.py가 만든 계산 결과)만 읽는다. */
(function () {
  'use strict';
  const D = window.AGENT_DATA;
  const $ = (s, el = document) => el.querySelector(s);
  if (!D) {
    $('#main').innerHTML = '<p class="card">데이터를 불러오지 못했습니다. data/agent_data.js 가 있는지 확인하세요.</p>';
    return;
  }

  const DOW = ['월', '화', '수', '목', '금', '토', '일'];
  const T = D.meta.peak_threshold;
  const dates = Object.keys(D.days).sort();
  const pad = (n) => String(n).padStart(2, '0');
  const hh = (h) => pad(h) + ':00';
  const fmt = (v, d = 1) => Number(v).toLocaleString('ko-KR', { minimumFractionDigits: d, maximumFractionDigits: d });
  const won = (n) => Math.round(n).toLocaleString('ko-KR') + '원';
  const md = (d) => `${+d.slice(5, 7)}/${+d.slice(8)}(${DOW[D.days[d].dow]})`;
  const maxOf = (a) => Math.max.apply(null, a);
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* 저장 불가 환경은 무시 */ } },
  };

  // ── 상태 ─────────────────────────────────────────────
  const m0 = /^r(\d+)_m(\d+)$/.exec(D.meta.default);
  const peakHoursOf = (d) => D.days[d].actual.filter((v) => v >= T).length;
  const S = {
    view: 'admin',
    date: dates.reduce((a, b) => (peakHoursOf(b) > peakHoursOf(a) ? b : a), dates[0]),
    share: +m0[1] / 100,
    margin: +m0[2],
    price: +(store.get('price') || 8000),
    week: 0,
    detail: null,
    pweek: D.plan_advice ? (Object.keys(D.plan_advice.weeks).find((k) => !D.plan_advice.weeks[k].summary.backtest && !/_ext$/.test(k)) || Object.keys(D.plan_advice.weeks)[0]) : null,
    pext: false,
  };
  const keyOf = () => `r${Math.round(S.share * 100)}_m${Math.round(S.margin)}`;
  const sc = () => D.scenarios[keyOf()];
  const adjOf = (d) => sc().days[d];

  function stats(d) {
    const raw = D.days[d], adj = adjOf(d);
    return {
      raw, adj,
      bMax: maxOf(raw.actual), aMax: maxOf(adj.actual_after),
      bPeak: raw.actual.filter((v) => v >= T).length,
      aPeak: adj.actual_after.filter((v) => v >= T).length,
      moved: adj.out.reduce((a, b) => a + b, 0),
    };
  }
  function hourState(d, h) {
    const raw = D.days[d], adj = adjOf(d);
    if (!raw.on[h]) return 'off';
    if (adj.out[h] > 0) return 'out';
    if (adj.in[h] > 0) return 'in';
    return 'normal';
  }
  function group(d, from, to) {
    const m = {};
    adjOf(d).transfers.forEach((t) => { (m[t[from]] = m[t[from]] || []).push(t); });
    return m;
  }
  const bySource = (d) => group(d, 'from');
  const byDest = (d) => group(d, 'to');
  const rangeText = (hs) => {
    const s = [...new Set(hs)].sort((a, b) => a - b), out = [];
    for (let i = 0; i < s.length;) {
      let j = i;
      while (j + 1 < s.length && s[j + 1] === s[j] + 1) j++;
      out.push(s[i] === s[j] ? hh(s[i]) : `${hh(s[i])}~${hh(s[j] + 1)}`);
      i = j + 1;
    }
    return out.join(', ');
  };
  // 3kW 미만의 미세한 이동은 현장에서 의미가 없어 근로자 화면에서는 '소폭 조정'으로 묶는다 (관리자 화면은 정확한 값을 그대로 보여 준다)
  const shiftTxt = (kw, sign) => (kw < 3 ? '소폭 조정' : `${sign}${fmt(kw, 0)}kW`);
  const live = (msg) => { $('#live').textContent = msg; };
  const acc = (title, body, open) => `<details class="acc"${open ? ' open' : ''}><summary>${title}</summary><div class="acc-body">${body}</div></details>`;
  const REDUCE = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const ease = (t) => 1 - Math.pow(1 - t, 3);
  // 숫자가 시작값에서 최종값까지 올라가거나 내려가는 효과 (동작 줄이기 설정이면 바로 최종값)
  function countUp(root) {
    root.querySelectorAll('.cu').forEach((el) => {
      const from = +el.dataset.from, to = +el.dataset.to, dec = +el.dataset.dec, suf = el.dataset.suf || '';
      const show = (v) => { el.textContent = Number(v).toLocaleString('ko-KR', { minimumFractionDigits: dec, maximumFractionDigits: dec }) + suf; };
      if (REDUCE || from === to) { show(to); return; }
      show(from);
      const t0 = performance.now(), dur = 900;
      const tick = (now) => { const k = Math.min(1, (now - t0) / dur); show(from + (to - from) * ease(k)); if (k < 1) requestAnimationFrame(tick); };
      requestAnimationFrame(tick);
      setTimeout(() => show(to), dur + 250); // 탭이 백그라운드라 프레임이 안 돌아도 최종값은 반드시 표시
    });
  }

  // ── 설정 ─────────────────────────────────────────────
  function initSettings() {
    const shares = D.meta.shares.filter((v) => v > 0);
    $('#s-share').innerHTML = shares.map((v) => `<option value="${v}">${Math.round(v * 100)}%${v === 0.1 ? ' (기본)' : ''}</option>`).join('');
    const mlabel = { 0: '0kW (예측 그대로)', 8: '8kW (기본 · 과거 편향 기준)', 15: '15kW (보수적)' };
    $('#s-margin').innerHTML = D.meta.margins.map((v) => `<option value="${v}">${mlabel[v] || v + 'kW'}</option>`).join('');
    $('#s-share').value = String(S.share);
    $('#s-margin').value = String(S.margin);
    $('#s-price').value = S.price;
    $('#s-share').onchange = (e) => { S.share = +e.target.value; S.detail = null; renderAll('이동 가능 비율을 바꿨습니다'); };
    $('#s-margin').onchange = (e) => { S.margin = +e.target.value; S.detail = null; renderAll('안전 마진을 바꿨습니다'); };
    $('#s-price').oninput = (e) => {
      const v = Math.max(0, +e.target.value || 0);
      S.price = v; store.set('price', v);
      renderKpis(false);
      if (S.view === 'plan' && PA) renderPlanKpis();
    };
  }

  // ── 관리자 ───────────────────────────────────────────
  function renderKpis(animate = true) {
    const s = sc().summary;
    const cut = s.period_max_before - s.period_max_after;
    const month = cut * S.price;
    const pct = s.peak_hours_before ? Math.round((1 - s.peak_hours_after / s.peak_hours_before) * 100) : 0;
    const maxDate = dates.reduce((a, b) => (maxOf(D.days[b].actual) > maxOf(D.days[a].actual) ? b : a));
    $('#kpis').innerHTML = `
      <div class="kpi"><div class="label">피크 시간 (${T}kW 이상)</div>
        <div class="value"><span class="from">${s.peak_hours_before}</span><span class="arrow">→</span><span class="cu" data-from="${s.peak_hours_before}" data-to="${s.peak_hours_after}" data-dec="0">${s.peak_hours_after}</span><span class="from">시간</span></div>
        <div class="delta">${pct}% 감소</div><div class="foot">실제 전력 기준 · ${dates.length}일</div></div>
      <div class="kpi"><div class="label">기간 최대수요 (요금 기준)</div>
        <div class="value"><span class="from">${fmt(s.period_max_before)}</span><span class="arrow">→</span><span class="cu" data-from="${s.period_max_before}" data-to="${s.period_max_after}" data-dec="1">${fmt(s.period_max_after)}</span><span class="from">kW</span></div>
        <div class="delta">${cut > 0 ? '−' + fmt(cut) + 'kW' : '변화 없음'}</div><div class="foot">최대는 ${md(maxDate)}에 발생</div></div>
      <div class="kpi"><div class="label">피크일의 일 최대 전력</div>
        <div class="value">−<span class="cu" data-from="0" data-to="${s.avg_daily_max_cut_on_peak_days}" data-dec="1">${fmt(s.avg_daily_max_cut_on_peak_days)}</span><span class="from">kW</span></div>
        <div class="delta">평균</div><div class="foot">피크가 있던 ${s.peak_days}일 기준</div></div>
      <div class="kpi"><div class="label">기본요금 절감 <span class="tag">예시 단가</span></div>
        <div class="value"><span class="cu" data-from="0" data-to="${Math.round(month)}" data-dec="0" data-suf="원">${won(month)}</span><span class="from">/월</span></div>
        <div class="delta">연 ${won(month * 12)}</div><div class="foot">최대수요 절감 × 단가 ${S.price.toLocaleString('ko-KR')}원/kW·월</div></div>`;    if (animate) countUp($('#kpis'));
  }

  function renderCallout() {
    const s = sc().summary;
    const maxDate = dates.reduce((a, b) => (maxOf(D.days[b].actual) > maxOf(D.days[a].actual) ? b : a));
    const act = maxOf(D.days[maxDate].actual), pred = maxOf(D.days[maxDate].pred);
    const cut = s.period_max_before - s.period_max_after;
    const m15 = D.scenarios[`r${Math.round(S.share * 100)}_m15`];
    let lead, more;
    if (cut < 5) {
      lead = `<strong>읽을 때 주의:</strong> 피크 시간은 ${s.peak_hours_before}→${s.peak_hours_after}시간으로 줄지만, 요금 기준 최대수요는 ${fmt(cut)}kW만 줄었습니다.`;
      more = `<p>요금을 정하는 기간 최대수요(${fmt(act, 0)}kW, ${md(maxDate)})는 M3 예측이 ${fmt(pred, 0)}kW로 ${fmt(act - pred, 0)}kW 낮게 보아 일정 조정 대상에서 빠졌습니다.</p>`;
    } else {
      lead = `<strong>참고:</strong> 기간 최대수요가 ${fmt(s.period_max_before)}→${fmt(s.period_max_after)}kW로 내려갔습니다.`;
      more = '<p>이 설정은 예측이 낮게 나온 시간까지 보수적으로 잡은 결과입니다.</p>';
    }
    if (S.margin < 15 && m15) {
      more += `<p>안전 마진을 15kW로 올리면 최대수요 ${fmt(m15.summary.period_max_after)}kW, 피크 ${m15.summary.peak_hours_after}시간까지 내려갑니다.
        15kW는 민감도 확인용이며 test를 보고 고른 값이 아닙니다. 기본 8kW는 과거 CV 편향 ${D.meta.margin_basis['2021-07']}·${D.meta.margin_basis['2021-08']}kW에서 왔습니다.</p>`;
    }
    $('#callout').innerHTML = `<p class="lead">${lead}</p>` + acc('왜 그런가요? 자세히 보기', more);
  }

  function chartSVG(d, anim) {
    const { raw, adj } = stats(d);
    const HALO = 'paint-order:stroke;stroke:var(--surface);stroke-width:4px;stroke-linejoin:round';
    const W = 760, H = 300, L = 46, R = 14, Tp = 16, B = 30;
    const iw = W - L - R, ih = H - Tp - B;
    const ymax = Math.max(220, Math.ceil((maxOf(raw.actual.concat([T])) + 10) / 20) * 20);
    const x = (h) => L + (iw * h) / 23, y = (v) => Tp + ih * (1 - v / ymax), bw = iw / 23;
    const pathOf = (arr) => arr.map((v, h) => `${h ? 'L' : 'M'}${x(h).toFixed(1)},${y(v).toFixed(1)}`).join('');
    const c = (name) => (anim ? ` class="${name}"` : '');
    const dl = (sec) => (anim ? `;animation-delay:${sec}s` : '');
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${md(d)} 시간별 전력. 변경 전 최대 ${fmt(maxOf(raw.actual), 0)}kW, 변경 후 최대 ${fmt(maxOf(adj.actual_after), 0)}kW. 표로 보기에서 자세한 값을 볼 수 있습니다.">`;
    for (let v = 0; v <= ymax; v += 50) {
      s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
    }
    for (let h = 0; h < 24; h++) {
      if (adj.out[h] > 0) s += `<rect${c('fade')} x="${x(h) - bw / 2}" y="${Tp}" width="${bw}" height="${ih}" style="fill:var(--out-bg)${dl(1.0)}"/>`;
      if (adj.in[h] > 0) s += `<rect${c('fade')} x="${x(h) - bw / 2}" y="${Tp}" width="${bw}" height="${ih}" style="fill:var(--in-bg)${dl(1.0)}"/>`;
    }
    for (let h = 0; h < 24; h += 3) s += `<text x="${x(h)}" y="${H - 10}" text-anchor="middle">${pad(h)}시</text>`;
    s += `<text class="axis-label" x="${L - 6}" y="10" text-anchor="end">kW</text>`;
    s += `<line x1="${L}" x2="${W - R}" y1="${y(T)}" y2="${y(T)}" style="stroke:var(--text-3)" stroke-width="1.2" stroke-dasharray="5 4"/>`;
    s += `<text style="${HALO}" x="${W - R}" y="${y(T) - 5}" text-anchor="end">피크 기준 ${fmt(T, 0)}kW</text>`;
    // 부하 이동 경로: 덜어낸 시간에서 받는 시간으로 호를 그리고 점이 따라간다 (그래프 아래쪽 빈 공간 사용)
    const y0 = y(0) - 6;
    adj.transfers.forEach((t, i) => {
      const x1 = x(t.from), x2 = x(t.to), yc = y0 - Math.min(70, 14 + Math.abs(x2 - x1) * 0.22);
      const dd = `M${x1.toFixed(1)},${y0} Q${((x1 + x2) / 2).toFixed(1)},${yc.toFixed(1)} ${x2.toFixed(1)},${y0}`;
      const delay = (0.5 + i * 0.08).toFixed(2);
      s += `<path${c('arc')} pathLength="1" d="${dd}" fill="none"${anim ? '' : ' opacity=".6"'} style="stroke:var(--text-3);stroke-width:${Math.min(4, 1 + t.kw / 6).toFixed(1)}${dl(delay)}"/>`;
      if (anim) {
        s += `<circle class="mv" r="3.5" style="fill:var(--text);animation-delay:${delay}s"><animateMotion dur="1s" begin="${delay}s" fill="freeze" path="${dd}"/></circle>`;
      }
    });
    s += `<path${c('draw')} pathLength="1" d="${pathOf(raw.actual)}" fill="none" style="stroke:var(--orange)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
    s += `<path id="p-after"${c('draw')} pathLength="1" d="${pathOf(anim ? raw.actual : adj.actual_after)}" fill="none" style="stroke:var(--blue)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
    raw.actual.forEach((v, h) => { if (v >= T) s += `<circle${c('fade')} cx="${x(h)}" cy="${y(v)}" r="4" style="fill:var(--orange);stroke:var(--surface)${dl(0.8)}" stroke-width="2"/>`; });
    const bi = raw.actual.indexOf(maxOf(raw.actual)), ai = adj.actual_after.indexOf(maxOf(adj.actual_after));
    if (maxOf(raw.actual) >= T) {
      s += `<text${c('fade')} style="${HALO}${dl(0.9)}" x="${x(bi)}" y="${y(maxOf(raw.actual)) - 9}" text-anchor="${bi > 18 ? 'end' : 'middle'}">변경 전 최대 ${fmt(maxOf(raw.actual), 0)}</text>`;
      if (adj.actual_after[ai] < maxOf(raw.actual) - 1) {
        s += `<circle${c('fade')} cx="${x(ai)}" cy="${y(adj.actual_after[ai])}" r="4" style="fill:var(--blue);stroke:var(--surface)${dl(2.3)}" stroke-width="2"/>`;
        s += `<text${c('fade')} style="${HALO}${dl(2.3)}" x="${x(ai) + (ai > 18 ? -9 : 9)}" y="${y(adj.actual_after[ai]) + 16}" text-anchor="${ai > 18 ? 'end' : 'start'}">변경 후 최대 ${fmt(adj.actual_after[ai], 0)}</text>`;
      }
    }
    s += `<line id="xh" x1="0" x2="0" y1="${Tp}" y2="${Tp + ih}" style="stroke:var(--text-3)" stroke-width="1" visibility="hidden"/>`;
    s += `<rect id="ov" x="${L}" y="${Tp}" width="${iw}" height="${ih}" fill="transparent"/></svg>`;
    return { s, W, L, iw, pathOf };
  }

  let chartRaf = 0, chartIO = null;
  function renderChart() {
    const d = S.date, { raw, adj, bMax, aMax, bPeak, aPeak } = stats(d);
    $('#a-day-summary').innerHTML = `<b>${md(d)}</b> · ${raw.day_type} · 최대 전력 ${fmt(bMax, 0)}kW → <b>${fmt(aMax, 0)}kW</b> · 피크 시간 ${bPeak} → <b>${aPeak}</b>시간`;
    $('#a-legend').innerHTML = `
      <span><i class="sw" style="background:var(--orange)"></i>변경 전 (실제 전력)</span>
      <span><i class="sw" style="background:var(--blue)"></i>변경 후 (조정 적용)</span>
      <span><i class="band" style="background:var(--out-bg);border:1px solid var(--line)"></i>부하를 덜어내는 시간</span>
      <span><i class="band" style="background:var(--in-bg);border:1px solid var(--line)"></i>부하를 받는 시간</span>
      <span><i class="sw" style="background:var(--text-3);height:2px"></i>부하 이동 경로</span>`;
    cancelAnimationFrame(chartRaf);
    if (chartIO) chartIO.disconnect();
    const anim = !REDUCE;
    const { s, W, L, iw, pathOf } = chartSVG(d, anim);
    const wrap = $('#a-chart');
    wrap.classList.remove('play');
    wrap.innerHTML = s + '<div class="tip" hidden></div>';
    const svg = $('svg', wrap), tip = $('.tip', wrap), xh = $('#xh', wrap);
    const move = (ev) => {
      const r = svg.getBoundingClientRect();
      const px = ((ev.clientX - r.left) / r.width) * W;
      const h = Math.max(0, Math.min(23, Math.round(((px - L) / iw) * 23)));
      const cx = L + (iw * h) / 23;
      xh.setAttribute('x1', cx); xh.setAttribute('x2', cx); xh.setAttribute('visibility', 'visible');
      const st = hourState(d, h);
      const stText = { out: `덜어냄 −${fmt(adj.out[h])}kW`, in: `받음 +${fmt(adj.in[h])}kW`, off: '비가동', normal: '유지' }[st];
      tip.innerHTML = `<b>${hh(h)} · ${stText}</b>
        <div class="row"><span>변경 전</span><span>${fmt(raw.actual[h], 0)}kW</span></div>
        <div class="row"><span>변경 후</span><span>${fmt(adj.actual_after[h], 0)}kW</span></div>
        <div class="row"><span>예측(M3)</span><span>${fmt(raw.pred[h], 0)}kW</span></div>
        <div class="row"><span>경보(M2)</span><span>${raw.m2[h] || '-'}</span></div>`;
      tip.hidden = false;
      const left = (cx / W) * r.width;
      tip.style.left = Math.min(Math.max(left + 12, 0), r.width - 170) + 'px';
      tip.style.top = '8px';
    };
    $('#ov', wrap).addEventListener('pointermove', move);
    $('#ov', wrap).addEventListener('pointerdown', move);
    $('#ov', wrap).addEventListener('pointerleave', () => { tip.hidden = true; xh.setAttribute('visibility', 'hidden'); });

    // 화면에 보일 때 재생: 선 그리기 → 부하 이동 경로 → 파란 선이 주황 선에서 내려옴
    const start = () => {
      wrap.classList.add('play');
      if (svg.unpauseAnimations) svg.unpauseAnimations();
      const el = $('#p-after', wrap), t0 = performance.now() + 1000, dur = 1300;
      const tick = (now) => {
        const k = Math.max(0, Math.min(1, (now - t0) / dur));
        el.setAttribute('d', pathOf(raw.actual.map((v, h) => v + (adj.actual_after[h] - v) * ease(k))));
        if (k < 1) chartRaf = requestAnimationFrame(tick);
      };
      chartRaf = requestAnimationFrame(tick);
      setTimeout(() => el.setAttribute('d', pathOf(adj.actual_after)), 1000 + 1300 + 250);
    };
    if (anim) {
      if (svg.pauseAnimations) svg.pauseAnimations();
      if ('IntersectionObserver' in window) {
        chartIO = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { chartIO.disconnect(); start(); } }, { threshold: 0.35 });
        chartIO.observe(wrap);
      } else start();
    } else {
      wrap.classList.add('play');
    }
    renderHourTable();
  }

  function renderHourTable() {
    const d = S.date, { raw, adj } = stats(d);
    const nm = { out: '덜어냄', in: '받음', off: '비가동', normal: '유지' };
    let rows = '';
    for (let h = 0; h < 24; h++) {
      rows += `<tr><td>${hh(h)}</td><td>${nm[hourState(d, h)]}</td><td class="num">${fmt(raw.pred[h], 0)}</td>
        <td class="num">${fmt(raw.actual[h], 0)}</td><td class="num">${fmt(adj.actual_after[h], 0)}</td><td>${raw.m2[h] || '-'}</td></tr>`;
    }
    $('#a-table').innerHTML = `<table><thead><tr><th>시간</th><th>일정</th><th class="num">예측(kW)</th><th class="num">변경 전(kW)</th><th class="num">변경 후(kW)</th><th>M2 경보</th></tr></thead><tbody>${rows}</tbody></table>`;
  }

  function reasonText(d, src) {
    const raw = D.days[d], adj = adjOf(d);
    const p = raw.pred[src], lv = raw.m2[src];
    let t = `이 시간의 예측 전력은 <span class="em">${fmt(p, 1)}kW</span>입니다. `;
    t += p >= T
      ? `피크 기준(${fmt(T, 0)}kW)을 넘을 것으로 예상됩니다. `
      : `기준(${fmt(T, 0)}kW)보다 조금 낮지만, 예측이 평소 ${S.margin}kW쯤 낮게 나오는 점을 더하면 ${fmt(p + S.margin, 1)}kW로 기준에 닿습니다. `;
    if (lv === '주의' || lv === '확정') t += `당일 실시간 경보(M2)도 이 시간을 '${lv}'으로 판단했습니다. `;
    t += `전기 요금은 가장 높았던 순간의 전력으로 정해져서, 이 시간의 부하 일부(${fmt(adj.out[src])}kW)를 여유 있는 시간으로 옮기면 최대수요를 낮출 수 있습니다.`;
    return t;
  }

  function renderChanges() {
    const d = S.date, { raw, adj, bMax } = stats(d);
    const src = bySource(d);
    const keys = Object.keys(src).map(Number).sort((a, b) => a - b);
    const ex = $('#a-expand');
    ex.hidden = !keys.length; ex.setAttribute('aria-pressed', 'false'); ex.textContent = '모두 펼치기';
    if (!keys.length) {
      $('#a-changes').innerHTML = bMax >= T
        ? `<p class="empty">이 날은 실제로 ${fmt(bMax, 0)}kW까지 올랐지만, 하루 전 예측으로는 피크 위험 시간을 찾지 못해 바꿀 일정이 없습니다. 이런 경우는 당일 실시간 경보(M2)로 대응해야 합니다.</p>`
        : `<p class="empty">이 날은 예측상 피크 위험 시간이 없어 바꿀 일정이 없습니다. (${raw.day_type})</p>`;
      return;
    }
    $('#a-changes').innerHTML = keys.map((h, idx) => {
      const dests = src[h];
      const destText = dests.map((t) => `${hh(t.to)}에 ${fmt(t.kw)}kW`).join(', ');
      const first = dests[0].to;
      return `<div class="change" style="--i:${idx}">
        <div class="change-head"><span class="when">${hh(h)} 부하 → ${rangeText(dests.map((t) => t.to))}</span>
          <span class="pill out">−${fmt(adj.out[h])}kW 덜어냄</span><span class="pill in">${destText}</span></div>
        <div class="ba">
          <div><div class="t">${hh(h)} 예측 전력 (변경 전 → 후)</div><div class="v">${fmt(raw.pred[h], 0)} → ${fmt(adj.pred_after[h], 0)}kW</div></div>
          <div class="arr">⇄</div>
          <div><div class="t">${hh(first)} 예측 전력 (변경 전 → 후)</div><div class="v">${fmt(raw.pred[first], 0)} → ${fmt(adj.pred_after[first], 0)}kW</div></div>
        </div>
        ${acc('왜 옮기나요? 자세히 보기', `<p class="why">${reasonText(d, h)}</p>`)}</div>`;
    }).join('');
  }

  function renderDays() {
    const rows = dates.map((d) => {
      const st = stats(d);
      return `<tr class="click${d === S.date ? ' sel' : ''}" data-d="${d}" tabindex="0">
        <td>${md(d)}</td><td>${st.raw.day_type}</td><td class="num">${fmt(st.bMax, 0)} → ${fmt(st.aMax, 0)}</td>
        <td class="num">${st.bPeak} → ${st.aPeak}</td><td class="num">${fmt(st.moved, 0)}</td></tr>`;
    }).join('');
    $('#a-days').innerHTML = `<table><thead><tr><th>날짜</th><th>날 유형(M1)</th><th class="num">일 최대 kW (전→후)</th><th class="num">피크 시간 (전→후)</th><th class="num">이동량 kW</th></tr></thead><tbody>${rows}</tbody></table>`;
    document.querySelectorAll('#a-days tr.click').forEach((tr) => {
      const go = () => { S.date = tr.dataset.d; renderAdmin(); $('#a-chart').scrollIntoView({ behavior: 'smooth', block: 'center' }); };
      tr.onclick = go;
      tr.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } };
    });
  }

  function renderMethod() {
    const b = D.meta.margin_basis, m = D.meta.models;
    $('#a-method').innerHTML = '<p class="muted">하루 전 예측으로 일정을 짜고, 같은 이동량을 그날의 실제 전력에 적용해 효과를 계산했습니다.</p>' +
      acc('어떻게 옮기나요?', `<ul>
        <li><b>덜어내는 시간</b>: 계획상 가동 중이고 (예측 + 안전 마진)이 ${fmt(T - 1, 0)}kW를 넘는 시간. 넘는 만큼만, 최대 이동 가능 비율까지.</li>
        <li><b>받는 시간</b>: 가동 중이고 덜어내지 않는 시간 중 예측이 낮은 곳부터. 받은 뒤에도 (예측 + 마진)이 ${fmt(T - 1 - D.meta.dest_buffer_kw, 0)}kW 이하가 되게 합니다. 같은 날, 최대 ±${D.meta.max_shift_h}시간 안에서만 옮깁니다.</li></ul>`) +
      acc('안전 마진은 어떻게 정했나요?', `<p>M3는 일 최대전력을 낮게 예측합니다(편향: 7월 ${b['2021-07']}, 8월 ${b['2021-08']}, test ${b.test}kW). 기본 ${D.meta.margins[1]}kW는 CV 두 달의 편향에서 정했고, test 결과를 보고 고르지 않았습니다.</p>`) +
      acc('예측은 얼마나 정확한가요?', `<ul><li>M3 test MAE ${m.m3_mae}kW, 일 최대 MAE ${m.m3_daily_max_mae}kW</li>
        <li>M2 주의 단계 정밀도 ${m.m2_alert.precision}, 재현율 ${m.m2_alert.recall}</li></ul>`) +
      acc('기본요금은 어떻게 계산하나요?', '<p>한국전력은 15분 단위 최대수요전력 중 가장 높은 값을 기준으로 요금을 부과합니다(가이드북). 절감액 = 최대수요 절감(kW) × 단가 × 12개월. 단가는 계약 조건마다 달라서 예시값을 직접 바꿔 입력하세요.</p>') +
      acc('반영하지 못한 것', '<p>작업별 이동 가능 여부, 점심·교대 시간, 인원 배치는 데이터에 없어 반영하지 않았습니다. 낮 부하를 새벽·야간으로 옮기면 야간 인건비(데이터의 인건비 비율 1.5)가 늘 수 있는데 이 비용도 계산에 넣지 않았습니다. 받는 시간이 실제로 작업 가능한지는 담당자가 확인해야 합니다.</p>') +
      acc('시뮬레이션 주의', `<p>${D.meta.notes.join(' ')}</p>`);
  }

  function renderAdmin() {
    $('#a-date').innerHTML = dates.map((d) => `<option value="${d}">${md(d)}${peakHoursOf(d) ? ' · 피크 ' + peakHoursOf(d) + '시간' : ''}</option>`).join('');
    $('#a-date').value = S.date;
    renderKpis(); renderCallout(); renderChart(); renderChanges(); renderDays(); renderMethod();
  }

  // ── 근로자 (에브리타임 스타일 시간표) ───────────────────
  const mondayOf = (d) => {
    const dt = new Date(d + 'T00:00:00Z');
    dt.setUTCDate(dt.getUTCDate() - ((dt.getUTCDay() + 6) % 7));
    return dt.toISOString().slice(0, 10);
  };
  const addDays = (d, n) => { const dt = new Date(d + 'T00:00:00Z'); dt.setUTCDate(dt.getUTCDate() + n); return dt.toISOString().slice(0, 10); };
  const weeks = [...new Set(dates.map(mondayOf))].sort();

  function runs(d) {
    const adj = adjOf(d), out = [];
    for (let h = 0; h < 24;) {
      const st = hourState(d, h);
      let e = h;
      while (e + 1 < 24 && hourState(d, e + 1) === st) e++;
      let kw = 0;
      for (let i = h; i <= e; i++) kw += st === 'out' ? adj.out[i] : st === 'in' ? adj.in[i] : 0;
      out.push({ d, st, s: h, e, kw });
      h = e + 1;
    }
    return out;
  }
  const NAME = { out: '고부하 작업 자제', in: '집중 배치', normal: '일반 가동', off: '비가동' };

  function renderWorker() {
    const mon = weeks[S.week];
    $('#w-range').textContent = `${md0(mon)} ~ ${md0(addDays(mon, 6))}`;
    $('#w-prev').disabled = S.week === 0;
    $('#w-next').disabled = S.week === weeks.length - 1;
    $('#w-legend').innerHTML = ['in', 'out', 'normal', 'off'].map((k) => `<span><i style="background:var(--${k === 'in' ? 'in' : k === 'out' ? 'out' : k === 'normal' ? 'normal' : 'off'}-bg)"></i>${NAME[k]}</span>`).join('');
    let html = '<div class="hd"></div>';
    const days7 = Array.from({ length: 7 }, (_, i) => addDays(mon, i));
    days7.forEach((d, i) => {
      const has = !!D.days[d];
      html += `<div class="hd${d === S.date ? ' sel' : ''}${has ? '' : ' dim'}">${DOW[i]}<small>${+d.slice(5, 7)}/${+d.slice(8)}</small></div>`;
    });
    html += '<div class="hours">' + Array.from({ length: 24 }, (_, h) => `<div>${pad(h)}</div>`).join('') + '</div>';
    days7.forEach((d, ci) => {
      if (!D.days[d]) { html += '<div class="col nodata">자료 없음</div>'; return; }
      html += '<div class="col">' + runs(d).map((r, bi) => {
        const len = r.e - r.s + 1;
        const sel = S.detail && S.detail.d === r.d && S.detail.s === r.s;
        const sub = r.st === 'out' ? shiftTxt(r.kw, '−') : r.st === 'in' ? shiftTxt(r.kw, '+') : '';
        return `<button type="button" class="blk ${r.st}" aria-pressed="${sel ? 'true' : 'false'}" data-d="${r.d}" data-s="${r.s}" data-e="${r.e}"
          style="--ci:${ci};--bi:${bi};top:calc(var(--rowh)*${r.s} + 1px);height:calc(var(--rowh)*${len} - 3px)"
          aria-label="${md(r.d)} ${hh(r.s)}부터 ${hh(r.e + 1)}까지 ${NAME[r.st]} ${sub}">
          <span class="nm">${NAME[r.st]}${sub ? ' ' + sub : ''}</span>${len > 1 ? `<span class="tm">${pad(r.s)}–${pad(r.e + 1)}시</span>` : ''}</button>`;
      }).join('') + '</div>';
    });
    $('#tt').innerHTML = html;
    document.querySelectorAll('#tt .blk').forEach((b) => {
      b.onclick = () => { S.detail = { d: b.dataset.d, s: +b.dataset.s, e: +b.dataset.e }; S.date = b.dataset.d; renderWorker(); };
    });
    renderWorkerDetail();
  }
  const md0 = (d) => `${+d.slice(5, 7)}/${+d.slice(8)}`;

  function defaultDetail() {
    const mon = weeks[S.week];
    let best = null;
    for (let i = 0; i < 7; i++) {
      const d = addDays(mon, i);
      if (!D.days[d]) continue;
      runs(d).forEach((r) => { if (r.st === 'out' && (!best || r.kw > best.kw)) best = r; });
    }
    return best ? { d: best.d, s: best.s, e: best.e } : null;
  }

  function renderWorkerDetail() {
    const box = $('#w-detail');
    if (!S.detail || !D.days[S.detail.d] || weeks[S.week] !== mondayOf(S.detail.d)) S.detail = defaultDetail();
    if (!S.detail) { box.innerHTML = '<p class="empty">이 주에는 전력이 몰리는 시간이 없습니다.</p>'; return; }
    const { d, s, e } = S.detail, st = hourState(d, s), raw = D.days[d], adj = adjOf(d);
    const hs = []; for (let h = s; h <= e; h++) hs.push(h);
    const when = `${md(d)} ${hh(s)}–${hh(e + 1)}`;
    const peak = Math.max.apply(null, hs.map((h) => raw.pred[h]));
    let lead, more = '';
    if (st === 'out') {
      const src = bySource(d), to = hs.flatMap((h) => (src[h] || []).map((t) => t.to));
      lead = `<p>전력이 몰리는 시간입니다. 옮길 시간: <b>${rangeText(to)}</b></p>`;
      more = `<p>이 시간대는 설비가 한꺼번에 돌아 전력이 가장 몰릴 것으로 예상됩니다 (예상 최대 <b>${fmt(peak, 0)}kW</b>, 기준 ${fmt(T, 0)}kW).</p>
        <p>큰 설비 가동이나 예열처럼 미룰 수 있는 작업을 옮겨 주세요. 총 ${fmt(hs.reduce((a, h) => a + adj.out[h], 0), 0)}kW만큼 덜어내는 것이 목표입니다.</p>`;
    } else if (st === 'in') {
      const dst = byDest(d), from = hs.flatMap((h) => (dst[h] || []).map((t) => t.from));
      lead = `<p>전력 여유가 있는 시간입니다. 작업을 가져올 시간: <b>${rangeText(from)}</b></p>`;
      more = `<p>예상 <b>${fmt(peak, 0)}kW</b>로 여유가 있어 미룬 작업을 여기에 배치할 수 있습니다. 최대 ${fmt(hs.reduce((a, h) => a + adj.in[h], 0), 0)}kW까지 받을 수 있습니다.</p>`;
    } else if (st === 'off') {
      lead = '<p>계획상 가동하지 않는 시간입니다.</p>';
    } else {
      lead = `<p>평소대로 가동하세요 (예상 최대 ${fmt(peak, 0)}kW).</p>`;
    }
    box.className = '';
    void box.offsetWidth; // 상세 카드가 바뀔 때마다 등장 효과를 다시 재생
    box.className = 'card detail';
    box.innerHTML = `<h3>${when} · ${NAME[st]}</h3>${lead}${more ? acc('이유 자세히 보기', more) : ''}
      <p class="small">관리자 화면과 같은 계산 결과입니다. 실제 작업 가능 여부는 현장 상황에 맞게 조정하세요.</p>`;
  }

  // ── 챗봇 ─────────────────────────────────────────────
  const log = $('#chat-log');
  function addMsg(who, html, acts) {
    const el = document.createElement('div');
    el.className = 'msg ' + who;
    if (who === 'user') el.textContent = html; else el.innerHTML = html;
    if (acts && acts.length) {
      const a = document.createElement('div'); a.className = 'acts';
      acts.forEach((x) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = x.label; b.onclick = x.fn; a.appendChild(b); });
      el.appendChild(a);
    }
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
  }
  const goto = (view, label) => ({ label, fn: () => setView(view) });

  function parseDate(t) {
    let m = /(\d{1,2})\s*(?:\/|월|\.)\s*(\d{1,2})/.exec(t);
    let d = m ? `2021-${pad(+m[1])}-${pad(+m[2])}` : null;
    if (!d) { m = /(\d{1,2})\s*일/.exec(t); d = m ? `2021-09-${pad(+m[1])}` : null; }
    return d && D.days[d] ? d : null;
  }
  const dayLine = (d) => `<b>${md(d)}</b> (${D.days[d].day_type})`;

  function ansPeak(d) {
    const src = bySource(d), hs = Object.keys(src).map(Number).sort((a, b) => a - b), raw = D.days[d];
    if (!hs.length) return `<p>${dayLine(d)}은 하루 전 예측으로는 피크 위험 시간이 없습니다.</p>`;
    const items = hs.map((h) => `<li>${hh(h)} · 예측 ${fmt(raw.pred[h], 0)}kW${raw.m2[h] === '정상' || !raw.m2[h] ? '' : ' · 경보 ' + raw.m2[h]}</li>`).join('');
    return `<p>${dayLine(d)}에 전력이 몰릴 시간: <b>${rangeText(hs)}</b></p>` + acc('시간별 예측 자세히 보기', `<p>피크 기준은 ${fmt(T, 0)}kW입니다.</p><ul>${items}</ul>`);
  }
  function ansSchedule(d) {
    const src = bySource(d), hs = Object.keys(src).map(Number).sort((a, b) => a - b);
    if (!hs.length) return `<p>${dayLine(d)}은 바꿀 일정이 없습니다. 평소대로 가동하세요.</p>`;
    const items = hs.map((h) => `<li>${hh(h)} 큰 작업 자제 (${shiftTxt(adjOf(d).out[h], '−')}) → ${rangeText(src[h].map((t) => t.to))}에 배치</li>`).join('');
    return `<p>${dayLine(d)} 큰 작업 자제 시간: <b>${rangeText(hs)}</b></p>` + acc('옮길 시간까지 자세히 보기', `<ul>${items}</ul>`);
  }
  function ansWhy(d) {
    const src = bySource(d), hs = Object.keys(src).map(Number);
    if (!hs.length) return `<p>${dayLine(d)}은 옮길 부하가 없어서 변경 이유도 없어요.</p>`;
    const h = hs.reduce((a, b) => (adjOf(d).out[b] > adjOf(d).out[a] ? b : a));
    return `<p>${dayLine(d)}에서 가장 크게 옮기는 ${hh(h)} 기준이에요.</p>` + acc('이유 자세히 보기', `<p>${reasonText(d, h)}</p>`, true);
  }
  function ansSummary() {
    const s = sc().summary, cut = s.period_max_before - s.period_max_after;
    return `<p>피크 시간 ${s.peak_hours_before} → <b>${s.peak_hours_after}</b>시간, 기간 최대수요 <b>−${fmt(cut)}kW</b>예요.</p>` +
      acc('금액과 이유 자세히 보기', `<p>현재 설정(이동 ${Math.round(S.share * 100)}%, 마진 ${S.margin}kW) 기준이에요.</p><ul>
        <li>기간 최대수요 ${fmt(s.period_max_before)} → ${fmt(s.period_max_after)}kW</li>
        <li>예시 단가 ${S.price.toLocaleString('ko-KR')}원/kW·월이면 월 <b>${won(cut * S.price)}</b>, 연 ${won(cut * S.price * 12)}</li></ul>
        <p>최대수요 절감이 작은 이유는 가장 높았던 날을 하루 전 예측이 놓쳤기 때문이에요.</p>`);
  }
  const ansLimits = () => '<p>시뮬레이션이라 실제 운영 결과와 다를 수 있어요.</p>' + acc('꼭 알아 둘 점 자세히 보기', `<ul>
      <li>2021-09-01~14 데이터로 되돌려 본 결과예요.</li>
      <li>하루 전 예측(M3)은 일 최대를 평균 8kW쯤 낮게 봐서, 그만큼 안전 마진을 더했어요.</li>
      <li>이동 가능 비율과 요금 단가는 데이터에 없어서 입력값이에요.</li>
      <li>작업별 이동 가능 여부, 점심·교대 시간은 반영하지 못했어요. 현장 확인이 필요해요.</li></ul>`);
  const ansHelp = () => `<p>이런 걸 물어보세요.</p><ul><li>"9/2 피크 위험 시간 알려줘"</li><li>"9/2 일정 추천해줘"</li><li>"왜 옮기는 거야?"</li><li>"절감 효과는?"</li><li>"한계가 뭐야?"</li></ul>`;

  function answer(text) {
    const t = text.replace(/\s+/g, '');
    const d = parseDate(text);
    if (d) { S.date = d; }
    const date = S.date;
    const acts = [goto('admin', '관리자 화면에서 보기'), goto('worker', '근로자 시간표 보기')];
    let html, a = acts;
    if (/안녕|도움|help|뭘할|무엇/i.test(t)) html = ansHelp(), a = [];
    else if (/한계|가정|신뢰|정확|믿|조심|주의할/.test(t)) html = ansLimits(), a = [goto('admin', '계산 방법 보기')];
    else if (/왜|이유|근거/.test(t)) html = ansWhy(date);
    else if (/절감|효과|요약|얼마|요금|단가|최대수요|돈/.test(t)) html = ansSummary(), a = [goto('admin', '관리자 화면에서 보기')];
    else if (/피크|위험|경보|몰리|언제/.test(t)) html = ansPeak(date);
    else if (/일정|추천|시간표|스케줄|배치|작업|근로|뭐해|어떻게/.test(t)) html = ansSchedule(date);
    else html = `<p>그 질문은 아직 몰라요. 아래 질문 중에서 골라 보세요.</p>` + ansHelp(), a = [];
    if (d) html = `<p class="small" style="margin:0 0 4px">날짜를 ${md(d)}로 바꿨어요.</p>` + html;
    addMsg('bot', html, a);
  }

  function initChat() {
    addMsg('bot', `<p>안녕하세요, 자원 최적화 에이전트예요. 피크 전력을 줄이는 일정 조정안을 설명해 드려요.</p><p>선택된 날짜는 ${dayLine(S.date)}예요. 날짜를 말하면 바꿔 드려요 (예: "9/10").</p>`, []);
    const chips = ['피크 위험 시간 알려줘', '일정 추천해줘', '왜 옮기는 거야?', '절감 효과는?', '한계가 뭐야?'];
    $('#chips').innerHTML = chips.map((c) => `<button type="button">${c}</button>`).join('');
    document.querySelectorAll('#chips button').forEach((b) => { b.onclick = () => ask(b.textContent); });
    $('#chat-form').onsubmit = (e) => { e.preventDefault(); const v = $('#chat-input').value.trim(); if (v) { $('#chat-input').value = ''; ask(v); } };
  }
  function ask(text) { addMsg('user', text); setTimeout(() => answer(text), 120); }

  // ── 주간 계획 (13 1주 전 예측 + 14 계획 수정 제안) ─────────
  const PA = D.plan_advice;
  const PW = PA ? PA.weeks : {};
  const pKey = () => S.pweek + (S.pext ? '_ext' : '');
  const pw = () => PW[pKey()];
  const dlabel = (d) => `${+d.slice(5, 7)}/${+d.slice(8, 10)}(${DOW[(new Date(d.slice(0, 10) + 'T00:00:00Z').getUTCDay() + 6) % 7]})`;
  const tlabel = (t) => `${dlabel(t)} ${t.slice(11, 13)}시`;
  const num0 = (v) => Math.round(v).toLocaleString('ko-KR');

  function initPlan() {
    if (!PA) return;
    const bases = [...new Set(Object.keys(PW).map((k) => k.replace(/_ext$/, '')))];
    const wname = (k) => {
      const w = PW[k], end = w.summary.end;
      return `${dlabel(w.start)} ~ ${dlabel(end)} · ${w.summary.backtest ? '백테스트 (실제 계획)' : '다음 주 (평소 일정 계획)'}`;
    };
    $('#p-week').innerHTML = bases.map((k) => `<option value="${k}">${wname(k)}</option>`).join('');
    $('#p-mode').innerHTML = `<option value="0">기본안: 생산량 이동 (최대 ${Math.round(PA.modes.base.max_move_share * 100)}%)</option>
      <option value="1">적극안: 최대 ${Math.round(PA.modes.ext.max_move_share * 100)}% + 가동 시간대 연장 (실험적)</option>`;
    $('#p-week').value = S.pweek; $('#p-mode').value = S.pext ? '1' : '0';
    $('#p-week').onchange = (e) => { S.pweek = e.target.value; renderPlan(); live('주를 바꿨습니다'); };
    $('#p-mode').onchange = (e) => { S.pext = e.target.value === '1'; renderPlan(); live('수정 범위를 바꿨습니다'); };
    $('#p-table-toggle').onclick = (e) => {
      const open = $('#p-table').hidden;
      $('#p-table').hidden = !open;
      e.target.setAttribute('aria-expanded', String(open));
      e.target.textContent = open ? '표 닫기' : '표로 보기';
    };
  }

  function renderPlanKpis() {
    const s = pw().summary;
    const cut = Math.max(0, s.pred_week_max_before - s.pred_week_max_after);
    const month = cut * S.price;
    const share = s.total_production ? (100 * s.moved_production / s.total_production) : 0;
    let k = `
      <div class="kpi"><div class="label">주간 예측 최대</div>
        <div class="value"><span class="from">${fmt(s.pred_week_max_before)}</span><span class="arrow">→</span><span class="cu" data-from="${s.pred_week_max_before}" data-to="${s.pred_week_max_after}" data-dec="1">${fmt(s.pred_week_max_after)}</span><span class="from">kW</span></div>
        <div class="delta">${cut > 0.05 ? '−' + fmt(cut) + 'kW' : '변화 없음'}</div><div class="foot">1주 전 예측 · 요금 기준이 되는 값</div></div>
      <div class="kpi"><div class="label">일 최대가 내려간 날</div>
        <div class="value">−<span class="cu" data-from="0" data-to="${s.avg_cut_on_lowered_days}" data-dec="1">${fmt(s.avg_cut_on_lowered_days)}</span><span class="from">kW</span></div>
        <div class="delta">${s.days_lowered}일 평균</div><div class="foot">받는 날은 최대 +${fmt(s.max_rise_on_receiving_days)}kW (기준보다 한참 아래)</div></div>
      <div class="kpi"><div class="label">옮긴 생산량</div>
        <div class="value"><span class="cu" data-from="0" data-to="${share}" data-dec="1" data-suf="%">${fmt(share)}%</span></div>
        <div class="delta">${num0(s.moved_production)} / 주간 ${num0(s.total_production)}</div><div class="foot">이동 ${s.n_moves}건${s.new_hours ? ` · 새로 켠 가동 ${s.new_hours}시간` : ''} · 총량은 그대로</div></div>`;
    if (s.backtest) {
      k += `<div class="kpi"><div class="label">실제 피크 시간 (${fmt(s.peak_threshold, 0)}kW 이상)</div>
        <div class="value"><span class="from">${s.actual_peak_hours}</span><span class="arrow">→</span><span class="cu" data-from="${s.actual_peak_hours}" data-to="${s.actual_peak_hours_after_est}" data-dec="0">${s.actual_peak_hours_after_est}</span><span class="from">시간</span></div>
        <div class="delta">추정</div><div class="foot">실제 전력 + 모델이 본 변화량</div></div>`;
    } else {
      k += `<div class="kpi"><div class="label">기본요금 절감 <span class="tag">예시 단가</span></div>
        <div class="value"><span class="cu" data-from="0" data-to="${Math.round(month)}" data-dec="0" data-suf="원">${won(month)}</span><span class="from">/월</span></div>
        <div class="delta">연 ${won(month * 12)}</div><div class="foot">주간 예측 최대 절감 × 단가 ${S.price.toLocaleString('ko-KR')}원/kW·월</div></div>`;
    }
    $('#p-kpis').innerHTML = k;
    countUp($('#p-kpis'));
  }

  function renderPlanChart() {
    const w = pw(), H0 = w.hours, n = w.time.length;
    const hasAct = !!(H0.actual && H0.actual.some((v) => v !== null));
    const W = 980, H = 320, L = 46, R = 14, Tp = 34, B = 28, iw = W - L - R, ih = H - Tp - B;
    const all = H0.pred_before.concat(H0.pred_after, hasAct ? H0.actual.filter((v) => v !== null) : [], [T]);
    const ymax = Math.ceil((maxOf(all) + 10) / 50) * 50;
    const x = (i) => L + (iw * i) / (n - 1), y = (v) => Tp + ih * (1 - v / ymax);
    const path = (a) => a.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('');
    const HALO = 'paint-order:stroke;stroke:var(--surface);stroke-width:4px;stroke-linejoin:round';
    const s0 = w.summary;
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${dlabel(w.start)}부터 7일 시간별 예측 전력. 원래 계획 최대 ${fmt(s0.pred_week_max_before, 0)}kW, 수정 계획 최대 ${fmt(s0.pred_week_max_after, 0)}kW. 표로 보기에서 자세한 값을 볼 수 있습니다.">`;
    for (let v = 0; v <= ymax; v += 50) s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
    // 가동을 새로 켠 시간 / 생산량을 덜어낸·받은 시간 표시
    const bw = iw / (n - 1);
    for (let i = 0; i < n; i++) {
      const dp = H0.production[i] - H0.production_before[i];
      if (H0.on_after[i] && !H0.on_before[i]) s += `<rect x="${x(i) - bw / 2}" y="${Tp}" width="${bw}" height="${ih}" style="fill:var(--in-bg);opacity:.9"/>`;
      else if (dp <= -1) s += `<rect x="${x(i) - bw / 2}" y="${y(0) - 8}" width="${bw}" height="8" style="fill:var(--orange);opacity:.55"/>`;
      else if (dp >= 1) s += `<rect x="${x(i) - bw / 2}" y="${y(0) - 8}" width="${bw}" height="8" style="fill:var(--blue);opacity:.55"/>`;
    }
    for (let dd = 0; dd < n / 24; dd++) {
      const xi = x(dd * 24);
      if (dd) s += `<line class="daysep" x1="${xi}" x2="${xi}" y1="${Tp - 22}" y2="${Tp + ih}"/>`;
      s += `<text class="dayname" x="${x(dd * 24 + 11.5)}" y="${Tp - 10}" text-anchor="middle">${dlabel(w.time[dd * 24])}</text>`;
    }
    s += `<text class="axis-label" x="${L - 6}" y="${Tp - 22}" text-anchor="end">kW</text>`;
    s += `<line x1="${L}" x2="${W - R}" y1="${y(T)}" y2="${y(T)}" style="stroke:var(--text-3)" stroke-width="1.2" stroke-dasharray="5 4"/>`;
    s += `<text style="${HALO}" x="${W - R}" y="${y(T) - 5}" text-anchor="end">피크 기준 ${fmt(T, 0)}kW</text>`;
    if (hasAct) s += `<path d="${path(H0.actual.map((v) => (v === null ? 0 : v)))}" fill="none" style="stroke:var(--text-3)" stroke-width="1.2" stroke-dasharray="2 3"/>`;
    s += `<path d="${path(H0.pred_before)}" fill="none" style="stroke:var(--orange)" stroke-width="2" stroke-linejoin="round"/>`;
    s += `<path d="${path(H0.pred_after)}" fill="none" style="stroke:var(--blue)" stroke-width="2" stroke-linejoin="round"/>`;
    s += `<line id="pxh" x1="0" x2="0" y1="${Tp}" y2="${Tp + ih}" style="stroke:var(--text-3)" stroke-width="1" visibility="hidden"/>`;
    s += `<circle id="pd1" r="4" style="fill:var(--orange);stroke:var(--surface)" stroke-width="2" visibility="hidden"/><circle id="pd2" r="4" style="fill:var(--blue);stroke:var(--surface)" stroke-width="2" visibility="hidden"/>`;
    s += `<rect id="pov" x="${L}" y="${Tp}" width="${iw}" height="${ih}" fill="transparent"/></svg>`;
    $('#p-legend').innerHTML = `
      <span><i class="sw" style="background:var(--orange)"></i>원래 계획의 예측</span>
      <span><i class="sw" style="background:var(--blue)"></i>수정 계획의 예측</span>
      ${hasAct ? '<span><i class="sw" style="background:var(--text-3)"></i>실제 전력 (점선, 비교용)</span>' : ''}
      <span><i class="band" style="background:var(--orange);opacity:.55"></i>생산량 덜어냄</span>
      <span><i class="band" style="background:var(--blue);opacity:.55"></i>생산량 받음</span>
      ${s0.new_hours ? '<span><i class="band" style="background:var(--in-bg);border:1px solid var(--line)"></i>새로 켠 가동 시간</span>' : ''}`;
    const wrap = $('#p-chart');
    wrap.classList.add('play');
    wrap.innerHTML = s + '<div class="tip" hidden></div>';
    const svg = $('svg', wrap), tip = $('.tip', wrap), xh = $('#pxh', wrap), d1 = $('#pd1', wrap), d2 = $('#pd2', wrap);
    const move = (ev) => {
      const r = svg.getBoundingClientRect();
      const px = ((ev.clientX - r.left) / r.width) * W;
      const i = Math.max(0, Math.min(n - 1, Math.round(((px - L) / iw) * (n - 1))));
      const cx = x(i);
      xh.setAttribute('x1', cx); xh.setAttribute('x2', cx); xh.setAttribute('visibility', 'visible');
      d1.setAttribute('cx', cx); d1.setAttribute('cy', y(H0.pred_before[i])); d1.setAttribute('visibility', 'visible');
      d2.setAttribute('cx', cx); d2.setAttribute('cy', y(H0.pred_after[i])); d2.setAttribute('visibility', 'visible');
      const onTxt = H0.on_after[i] && !H0.on_before[i] ? '새로 가동' : H0.on_before[i] ? '가동' : '비가동';
      tip.innerHTML = `<b>${tlabel(w.time[i])} · ${onTxt}</b>
        <div class="row"><span>원래 계획 예측</span><span>${fmt(H0.pred_before[i], 0)}kW</span></div>
        <div class="row"><span>수정 계획 예측</span><span>${fmt(H0.pred_after[i], 0)}kW</span></div>
        <div class="row"><span>생산량</span><span>${num0(H0.production_before[i])} → ${num0(H0.production[i])}</span></div>
        ${hasAct && H0.actual[i] !== null ? `<div class="row"><span>실제 전력</span><span>${fmt(H0.actual[i], 0)}kW</span></div>` : ''}`;
      tip.hidden = false;
      const left = (cx / W) * r.width;
      tip.style.left = (left > r.width - 200 ? left - 190 : left + 12) + 'px';
      tip.style.top = '8px';
    };
    $('#pov', wrap).addEventListener('pointermove', move);
    $('#pov', wrap).addEventListener('pointerdown', move);
    $('#pov', wrap).addEventListener('pointerleave', () => { tip.hidden = true; [xh, d1, d2].forEach((e) => e.setAttribute('visibility', 'hidden')); });

    let rows = '';
    for (let i = 0; i < n; i++) {
      rows += `<tr><td>${tlabel(w.time[i])}</td><td>${H0.on_before[i] ? '가동' : '비가동'}${H0.on_after[i] !== H0.on_before[i] ? ' → 가동' : ''}</td>
        <td class="num">${num0(H0.production_before[i])} → ${num0(H0.production[i])}</td>
        <td class="num">${fmt(H0.pred_before[i], 0)}</td><td class="num">${fmt(H0.pred_after[i], 0)}</td>
        ${hasAct ? `<td class="num">${H0.actual[i] === null ? '-' : fmt(H0.actual[i], 0)}</td>` : ''}</tr>`;
    }
    $('#p-table').innerHTML = `<table><thead><tr><th>시간</th><th>가동</th><th class="num">생산량 (전→후)</th><th class="num">원래 예측(kW)</th><th class="num">수정 예측(kW)</th>${hasAct ? '<th class="num">실제(kW)</th>' : ''}</tr></thead><tbody>${rows}</tbody></table>`;
  }

  function renderPlanMoves() {
    const w = pw(), days = {};
    w.days.forEach((r) => { days[r.date] = r; });
    if (!w.moves.length) { $('#p-moves').innerHTML = '<p class="empty">규칙 안에서 최대전력을 의미 있게 낮추는 이동을 찾지 못했습니다.</p>'; return; }
    const ordered = w.moves.slice().sort((a, b) => (a.kind === b.kind ? 0 : a.kind === 'day' ? -1 : 1));
    $('#p-moves').innerHTML = ordered.map((m, i) => {
      const f = m.from.slice(0, 10), t = m.to.slice(0, 10), fd = days[f], td = days[t];
      if (m.kind === 'day') {
        const pct = fd.prod_before ? Math.round(100 * m.qty / fd.prod_before) : 0;
        const ext = m.new_hours.length ? `<p class="why"><span class="tag exp">실험적</span> ${dlabel(t)} ${m.new_hours.map((h) => h.slice(-3)).join(', ')}를 새로 가동해서 받습니다. 과거에 이 요일·시각에 가동한 선례가 있는 시간만 골랐지만, 선례가 적어 예측을 그대로 믿기 어렵습니다.</p>` : '';
        return `<div class="change" style="--i:${i}">
          <div class="change-head"><span class="when">${dlabel(f)} → ${dlabel(t)}</span>
            <span class="pill out">−${num0(m.qty)} 생산량 (${dlabel(f)} 하루의 ${pct}%)</span><span class="pill in">+${num0(m.qty)} 받음</span></div>
          <div class="ba">
            <div><div class="t">${dlabel(f)} 예측 최대 (전 → 후)</div><div class="v">${fmt(fd.pred_max_before, 0)} → ${fmt(fd.pred_max_after, 0)}kW</div></div>
            <div class="arr">⇄</div>
            <div><div class="t">${dlabel(t)} 예측 최대 (전 → 후)</div><div class="v">${fmt(td.pred_max_before, 0)} → ${fmt(td.pred_max_after, 0)}kW</div></div>
          </div>${ext}
          ${acc('왜 다른 날로 옮기나요?', `<p class="why">1주 전 예측 모델은 시간별 생산량보다 <span class="em">그날 총 생산량</span>에 더 크게 반응합니다. 그래서 같은 날 안에서 시간을 바꾸는 것보다, 최대전력이 높은 날의 물량 일부를 여유 있는 날로 넘기는 쪽이 일 최대를 더 낮춥니다. 덜어내는 시간은 그날 예측이 높은 낮 시간 위주이고, 받는 날은 예측이 낮은 가동 시간부터 채웁니다.</p>`)}</div>`;
      }
      return `<div class="change" style="--i:${i}">
        <div class="change-head"><span class="when">${tlabel(m.from)} → ${m.to.slice(11, 13)}시 (같은 날)</span>
          <span class="pill out">−${num0(m.qty)} 생산량</span><span class="pill in">+${num0(m.qty)} 받음</span></div></div>`;
    }).join('');
  }

  function renderPlanDays() {
    const w = pw(), bt = w.summary.backtest;
    const rows = w.days.map((r) => `<tr><td>${dlabel(r.date)}</td><td class="num">${r.on_before} → ${r.on_after}</td>
      <td class="num">${num0(r.prod_before)} → ${num0(r.prod_after)}</td><td class="num">${fmt(r.pred_max_before, 0)} → ${fmt(r.pred_max_after, 0)}</td>
      <td class="num">${r.risk_before} → ${r.risk_after}</td>${bt ? `<td class="num">${fmt(r.actual_max, 0)} → ${fmt(r.actual_max_after_est, 0)}</td>` : ''}</tr>`).join('');
    $('#p-days').innerHTML = `<table><thead><tr><th>날짜</th><th class="num">가동 시간</th><th class="num">생산량 (전→후)</th><th class="num">예측 최대 kW</th><th class="num">위험 시간</th>${bt ? '<th class="num">실제 최대 → 추정</th>' : ''}</tr></thead><tbody>${rows}</tbody></table>`;
  }

  function renderPlanMethod() {
    const m = PA.model;
    $('#p-method').innerHTML = '<p class="muted">계획(시간별 가동 여부 + 생산량)을 넣으면 1주 전 예측 모델로 전력을 예측하고, 일 최대전력이 낮아지도록 생산량을 옮기는 안을 하나씩 모델로 다시 예측해 골랐습니다. 주간 총 생산량은 바꾸지 않습니다.</p>' +
      acc('예측은 얼마나 정확한가요?', `<ul><li>1주 전 예측 test(9/1~14) MAE ${fmt(m.test_mae, 2)}kW, 일 최대 MAE ${fmt(m.test_daily_max_mae, 2)}kW (하루 전 M3는 ${D.meta.models.m3_mae}kW)</li>
        <li>7·8월 CV MAE ${fmt(m.cv_mae, 2)}kW. 계획을 넣지 않으면 23kW대로 커집니다(8월 하계 휴무를 모름).</li>
        <li>위험 시간 = 예측 + 안전 마진 ${fmt(m.margin, 0)}kW ≥ ${fmt(T, 0)}kW. 마진은 CV에서 경보 F1이 가장 높은 값입니다.</li></ul>`) +
      acc('기본안과 적극안의 차이', `<ul><li><b>기본안</b>: 생산량만 옮깁니다. 시간·하루마다 원래 생산량의 ${Math.round(PA.modes.base.max_move_share * 100)}%까지.</li>
        <li><b>적극안</b>: ${Math.round(PA.modes.ext.max_move_share * 100)}%까지 옮기고, 받는 날 가동 구간 끝에 붙은 비가동 시간을 켤 수 있습니다. 과거 같은 요일·시각에 5% 이상 가동한 선례가 있는 시간(예: 토요일 낮, 월요일 새벽)만 허용합니다. 평일 낮을 끄는 안은 데이터에 선례가 없어 만들지 않습니다.</li></ul>`) +
      acc('한계', `<ul><li>이 데이터에서 평일 낮 전력은 생산량보다 가동 자체에 크게 좌우됩니다. 생산량을 옮겨 줄이는 폭은 날마다 몇~10kW 수준이고, 남는 피크는 당일 M2 경보로 대응해야 합니다.</li>
        <li>받는 날(주로 토요일 새벽)에 물량이 몰리면 야간 인건비(1.5배)와 인력 배치 문제가 생길 수 있는데 계산에 넣지 않았습니다.</li>
        <li>백테스트의 '실제 피크 추정'은 실제 전력에 모델이 본 변화량을 더한 값이라, 실제로 그렇게 운영했을 때의 결과는 아닙니다.</li></ul>`) +
      acc('내 계획을 넣어 보려면', `<p>웹은 미리 계산한 결과만 보여 줍니다. 직접 계획을 넣으려면 저장소에서 실행하세요.</p><ul>
        <li><code>python 14_plan_advisor.py --template 2021-09-15</code> → plans/plan_2021-09-15.csv</li>
        <li>CSV의 on(가동 1/0)과 production(생산량)을 고친 뒤 <code>python 14_plan_advisor.py --plan plans/plan_2021-09-15.csv</code></li>
        <li>결과: plans/out/…/advice.md, revised_plan.csv (수정된 계획)</li></ul>`);
  }

  function renderPlan() {
    if (!PA) {
      $('#view-plan').innerHTML = '<p class="card">주간 계획 데이터가 없습니다. 13_week_ahead.py → 14_plan_advisor.py --web-examples → 12_build_report.py 순서로 실행하세요.</p>';
      return;
    }
    const w = pw(), s = w.summary;
    $('#p-lead').innerHTML = s.backtest
      ? `${dlabel(s.start)}~${dlabel(s.end)}에 <b>실제로 실행된 계획</b>을 넣어 본 백테스트입니다. 예측에는 ${dlabel(s.start)} 이전 전력만 썼고, 실제 전력은 비교용입니다.`
      : `데이터 다음 주(${dlabel(s.start)}~${dlabel(s.end)})의 <b>평소 일정 계획</b>(요일별 가동 + 최근 4주 평균 생산량)을 넣은 예시입니다. 실제 계획 CSV를 넣으면 같은 방식으로 수정안이 나옵니다.`;
    renderPlanKpis(); renderPlanChart(); renderPlanMoves(); renderPlanDays(); renderPlanMethod();
  }

  // ── 공통 ─────────────────────────────────────────────
  function setView(v) {
    S.view = v;
    ['admin', 'worker', 'plan', 'chat'].forEach((k) => {
      $('#view-' + k).hidden = k !== v;
      $('#tab-' + k).setAttribute('aria-selected', String(k === v));
    });
    document.querySelectorAll('.settings .m4-only').forEach((el) => { el.hidden = v === 'plan'; });
    if (v === 'admin') renderAdmin();
    if (v === 'plan') renderPlan();
    if (v === 'worker') { S.week = Math.max(0, weeks.indexOf(mondayOf(S.date))); S.detail = null; renderWorker(); }
    try { history.replaceState(null, '', '#' + v); } catch (e) { /* 무시 */ }
    window.scrollTo({ top: 0 });
  }
  function renderAll(msg) {
    if (S.view === 'admin') renderAdmin();
    if (S.view === 'worker') renderWorker();
    if (msg) live(msg + ' · ' + sc().summary.peak_hours_before + '→' + sc().summary.peak_hours_after + '시간');
  }

  function init() {
    initSettings();
    document.querySelectorAll('.tabs button').forEach((b) => { b.onclick = () => setView(b.dataset.view); });
    $('#a-date').onchange = (e) => { S.date = e.target.value; renderAdmin(); };
    $('#a-table-toggle').onclick = (e) => {
      const open = $('#a-table').hidden;
      $('#a-table').hidden = !open;
      e.target.setAttribute('aria-expanded', String(open));
      e.target.textContent = open ? '표 닫기' : '표로 보기';
    };
    $('#a-replay').onclick = () => renderChart();
    $('#a-expand').onclick = (e) => {
      const open = e.target.getAttribute('aria-pressed') !== 'true';
      document.querySelectorAll('#a-changes details').forEach((x) => { x.open = open; });
      e.target.setAttribute('aria-pressed', String(open));
      e.target.textContent = open ? '모두 접기' : '모두 펼치기';
    };
    $('#w-prev').onclick = () => { if (S.week > 0) { S.week--; S.detail = null; renderWorker(); } };
    $('#w-next').onclick = () => { if (S.week < weeks.length - 1) { S.week++; S.detail = null; renderWorker(); } };
    $('#theme').onclick = () => {
      const cur = document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      const next = cur === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = next; store.set('theme', next);
    };
    const th = store.get('theme'); if (th) document.documentElement.dataset.theme = th;
    initChat();
    initPlan();
    const h = location.hash.slice(1);
    setView(['admin', 'worker', 'plan', 'chat'].includes(h) ? h : 'admin');
    window.addEventListener('hashchange', () => {
      const n = location.hash.slice(1);
      if (['admin', 'worker', 'plan', 'chat'].includes(n) && n !== S.view) setView(n);
    });
  }
  init();
})();
