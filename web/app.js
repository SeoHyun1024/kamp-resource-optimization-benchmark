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
  const dowOf = (d) => (D.days[d] ? D.days[d].dow : (new Date(d.slice(0, 10) + 'T00:00:00Z').getUTCDay() + 6) % 7);
  const md = (d) => `${+d.slice(5, 7)}/${+d.slice(8, 10)}(${DOW[dowOf(d)]})`;
  const maxOf = (a) => Math.max.apply(null, a);
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* 저장 불가 환경은 무시 */ } },
  };

  // ── 상태 ─────────────────────────────────────────────
  const m0 = /^r(\d+)_m(\d+)$/.exec(D.meta.default);
  const peakHoursOf = (d) => D.days[d].actual.filter((v) => v >= T).length;
  // 기본요금 단가: tariff.json -> 12_build_report.py -> agent_data.js (단가가 바뀌면 tariff.json만 고침)
  const TARIFF = D.meta.tariff;
  const tariffTag = () => `<span class="tag" title="${TARIFF.label}">추정 종별</span>`;
  // 요금적용전력은 직전 12개월 최대수요라 한 주를 낮춰도 바로 줄지 않는다 -> "12개월 유지 시 추정"
  const tariffFoot = (what) => `${what} × ${S.price.toLocaleString('ko-KR')}원 · 12개월 유지 시 추정` +
    (S.price === TARIFF.price ? ` (${TARIFF.basis.replace(' · ', ', ')})` : ' (입력한 단가)');
  const S = {
    view: 'admin',
    date: dates.reduce((a, b) => (peakHoursOf(b) > peakHoursOf(a) ? b : a), dates[0]),
    share: +m0[1] / 100,
    margin: +m0[2],
    price: +(store.get('price') || TARIFF.price),
    week: 0,
    detail: null,
    pweek: null,  // 주간 계획 탭의 '주' (group). initPlan에서 정함
    aweek: 0,     // 관리자 탭 주간 피드백의 주 (index). initWeeks 전에 마지막 주로 맞춤
    pext: false,
    showMove: store.get('showMove') === '1',  // 하루 그래프: 부하 이동 경로 겹쳐 보기
    showHs: store.get('showHs') === '1',      // 하루 그래프: 사후 최선 겹쳐 보기
  };
  if (D.weeks) {   // 주간 피드백이 있으면 마지막 주, 그 주에서 피크가 가장 많은 날로 시작
    S.aweek = D.weeks.length - 1;
    const ds = D.weeks[S.aweek].days.filter((d) => D.days[d]);
    S.date = ds.reduce((a, b) => (peakHoursOf(b) > peakHoursOf(a) ? b : a), ds[0]);
  }
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
  // 가로로 스크롤되는 날짜 버튼 줄에서 고른 날이 보이게 (페이지 세로 스크롤은 건드리지 않음)
  const showPressed = (row) => {
    const b = $('[aria-pressed="true"]', row);
    if (b && row.scrollWidth > row.clientWidth) row.scrollLeft = b.offsetLeft - row.offsetLeft - (row.clientWidth - b.offsetWidth) / 2;
  };
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
    $('#s-price-tag').title = TARIFF.label;
    $('#tariff-note').textContent = `기본요금 단가 기본값 ${TARIFF.price.toLocaleString('ko-KR')}${TARIFF.unit}은 ${TARIFF.label.replace(' (추정)', '')}로 추정한 값입니다(${TARIFF.basis.replace(' · ', ', ')}).`;
    $('#s-share').onchange = (e) => { S.share = +e.target.value; S.detail = null; renderAll('이동 가능 비율을 바꿨습니다'); };
    $('#s-margin').onchange = (e) => { S.margin = +e.target.value; S.detail = null; renderAll('안전 마진을 바꿨습니다'); };
    $('#s-price').oninput = (e) => {
      const v = Math.max(0, +e.target.value || 0);
      S.price = v; store.set('price', v);
      if (S.view === 'admin') { if (WK) renderWeekKpis(false); else renderKpis(false); }   // 주간 피드백이면 주간 카드
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
      <div class="kpi"><div class="label">기본요금 절감 ${tariffTag()}</div>
        <div class="value"><span class="cu" data-from="0" data-to="${Math.round(month)}" data-dec="0" data-suf="원">${won(month)}</span><span class="from">/월</span></div>
        <div class="delta">연 ${won(month * 12)}</div><div class="foot">${tariffFoot('최대수요 절감')}</div></div>`;    if (animate) countUp($('#kpis'));
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
    const W = 760, H = 290, L = 46, R = 14, Tp = 16, B = 44;
    const iw = W - L - R, ih = H - Tp - B;
    // y축은 그날 값 근처만 보여 준다 (0부터 그리면 변경 전·후 선이 위쪽에 붙어 차이가 안 보임)
    const hs0 = S.showHs ? hsOf(d) : null;
    const all = raw.actual.concat(adj.actual_after, hs0 ? hs0.actual_after : [], [T]);
    const ymax = Math.ceil((maxOf(all) + 8) / 10) * 10;
    const ymin = Math.max(0, Math.floor((Math.min.apply(null, all) - 10) / 25) * 25);
    const ystep = ymax - ymin > 120 ? 50 : 25;
    const x = (h) => L + (iw * h) / 23, y = (v) => Tp + ih * (1 - (v - ymin) / (ymax - ymin)), bw = iw / 23;
    const pathOf = (arr) => arr.map((v, h) => `${h ? 'L' : 'M'}${x(h).toFixed(1)},${y(v).toFixed(1)}`).join('');
    const c = (name) => (anim ? ` class="${name}"` : '');
    const dl = (sec) => (anim ? `;animation-delay:${sec}s` : '');
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${md(d)} 시간별 전력. 변경 전 최대 ${fmt(maxOf(raw.actual), 0)}kW, 변경 후 최대 ${fmt(maxOf(adj.actual_after), 0)}kW. 표로 보기에서 자세한 값을 볼 수 있습니다.">`;
    s += `<defs><clipPath id="over-t"><rect x="${L}" y="${Tp}" width="${iw}" height="${Math.max(0, y(T) - Tp)}"/></clipPath></defs>`;
    for (let v = Math.ceil(ymin / ystep) * ystep; v <= ymax; v += ystep) {
      s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
    }
    // 부하 이동 경로를 켜면 덜어내는/받는 시간을 그래프 전체 높이로 칠한다
    if (S.showMove) {
      for (let h = 0; h < 24; h++) {
        if (adj.out[h] > 0) s += `<rect${c('fade')} x="${x(h) - bw / 2}" y="${Tp}" width="${bw}" height="${ih}" style="fill:var(--out-bg)${dl(1.0)}"/>`;
        if (adj.in[h] > 0) s += `<rect${c('fade')} x="${x(h) - bw / 2}" y="${Tp}" width="${bw}" height="${ih}" style="fill:var(--in-bg)${dl(1.0)}"/>`;
      }
    }
    // x축 바로 아래 띠: 덜어낸 시간(주황) / 받은 시간(파랑)
    const sy = Tp + ih + 4;
    for (let h = 0; h < 24; h++) {
      const col = adj.out[h] > 0 ? 'var(--orange)' : adj.in[h] > 0 ? 'var(--blue)' : null;
      if (col) s += `<rect${c('fade')} x="${x(h) - bw / 2 + 1}" y="${sy}" width="${bw - 2}" height="6" rx="2" style="fill:${col}${dl(1.0)}"><title>${hh(h)} ${adj.out[h] > 0 ? '덜어냄 −' + fmt(adj.out[h]) : '받음 +' + fmt(adj.in[h])}kW</title></rect>`;
    }
    for (let h = 0; h < 24; h += 3) s += `<text x="${x(h)}" y="${H - 8}" text-anchor="middle">${pad(h)}시</text>`;
    s += `<text class="axis-label" x="${L - 6}" y="10" text-anchor="end">kW</text>`;
    // 피크 기준을 넘은 부분(변경 전)을 주황으로 칠해 '어디가 문제였는지'를 먼저 보이게 한다
    s += `<path${c('fade')} clip-path="url(#over-t)" d="${pathOf(raw.actual)}L${x(23)},${y(ymin)}L${x(0)},${y(ymin)}Z" style="fill:var(--orange);opacity:.22${dl(0.8)}"/>`;
    s += `<line x1="${L}" x2="${W - R}" y1="${y(T)}" y2="${y(T)}" style="stroke:var(--text-3)" stroke-width="1.2" stroke-dasharray="5 4"/>`;
    s += `<text style="${HALO}" x="${W - R}" y="${y(T) + 15}" text-anchor="end">피크 기준 ${fmt(T, 0)}kW</text>`;
    // 부하 이동 경로: 덜어낸 시간에서 받는 시간으로 호를 그리고 점이 따라간다 (그래프 아래쪽에서)
    const y0 = Tp + ih - 2;
    if (S.showMove) adj.transfers.forEach((t, i) => {
      const x1 = x(t.from), x2 = x(t.to), yc = y0 - Math.min(70, 14 + Math.abs(x2 - x1) * 0.22);
      const dd = `M${x1.toFixed(1)},${y0} Q${((x1 + x2) / 2).toFixed(1)},${yc.toFixed(1)} ${x2.toFixed(1)},${y0}`;
      const delay = (0.5 + i * 0.08).toFixed(2);
      s += `<path${c('arc')} pathLength="1" d="${dd}" fill="none"${anim ? '' : ' opacity=".6"'} style="stroke:var(--text-3);stroke-width:${Math.min(4, 1 + t.kw / 6).toFixed(1)}${dl(delay)}"/>`;
      if (anim) {
        s += `<circle class="mv" r="3.5" style="fill:var(--text);animation-delay:${delay}s"><animateMotion dur="1s" begin="${delay}s" fill="freeze" path="${dd}"/></circle>`;
      }
    });
    s += `<path${c('draw')} pathLength="1" d="${pathOf(raw.actual)}" fill="none" style="stroke:var(--orange)" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>`;
    const hsD = hs0;
    if (hsD) s +=`<path${c('fade')} d="${pathOf(hsD.actual_after)}" fill="none" style="stroke:var(--text-3)${dl(2.4)}" stroke-width="1.6" stroke-dasharray="5 4"/>`;
    s += `<path id="p-after"${c('draw')} pathLength="1" d="${pathOf(anim ? raw.actual : adj.actual_after)}" fill="none" style="stroke:var(--blue)" stroke-width="2.75" stroke-linejoin="round" stroke-linecap="round"/>`;
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
    const d = S.date, { raw, adj, bMax, aMax, bPeak, aPeak, moved } = stats(d);
    $('#a-day-summary').innerHTML = `
      <div class="stat"><span class="k">${md(d)}</span><span class="v">${raw.day_type}</span></div>
      <div class="stat"><span class="k">최대 전력</span><span class="v">${fmt(bMax, 0)} → <b>${fmt(aMax, 0)}</b>kW ${badge(bMax - aMax, 'kW')}</span></div>
      <div class="stat"><span class="k">피크 시간</span><span class="v">${bPeak} → <b>${aPeak}</b>시간 ${badge(bPeak - aPeak, '시간')}</span></div>
      <div class="stat"><span class="k">옮긴 부하</span><span class="v"><b>${fmt(moved, 0)}</b>kW</span></div>`;
    $('#a-show-hs-wrap').hidden = !hsOf(d);
    $('#a-legend').innerHTML = `
      <span><i class="sw" style="background:var(--orange)"></i>변경 전 (실제)</span>
      <span><i class="sw" style="background:var(--blue)"></i>변경 후 (AI 계획 적용)</span>
      <span><i class="band" style="background:var(--orange);opacity:.3"></i>피크 기준 초과</span>
      <span><i class="tick" style="background:var(--orange)"></i>덜어낸 시간</span>
      <span><i class="tick" style="background:var(--blue)"></i>받은 시간</span>
      ${S.showMove ? '<span><i class="sw" style="background:var(--text-3);height:2px"></i>부하 이동 경로</span>' : ''}
      ${S.showHs && hsOf(d) ? '<span><i class="sw dash"></i>사후 최선 (예측이 완벽했다면)</span>' : ''}`;
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
      // 한 줄 = 옮기는 한 건. 누르면 이유가 펼쳐진다
      return `<details class="change" style="--i:${idx}"><summary class="change-row">
          <span class="when">${hh(h)} <span class="to">→ ${rangeText(dests.map((t) => t.to))}</span></span>
          <span class="pill out">−${fmt(adj.out[h])}kW</span>
          <span class="mv-pred"><span class="t">${hh(h)} 예측</span> ${fmt(raw.pred[h], 0)} → <b>${fmt(adj.pred_after[h], 0)}</b></span>
          <span class="mv-pred"><span class="t">${hh(first)} 예측</span> ${fmt(raw.pred[first], 0)} → <b>${fmt(adj.pred_after[first], 0)}</b>kW</span>
          <span class="why-btn">왜?</span></summary>
        <p class="why">${reasonText(d, h)}${dests.length > 1 ? ` 받는 시간: ${destText}.` : ''}</p></details>`;
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
    const fbAcc = WK ? acc('주마다 다시 학습한다는 게 무슨 뜻인가요?', `<ul>
        <li>각 주는 <b>그 주가 시작되기 전날까지</b>의 데이터만으로 하루 전 예측 모델(M3)을 다시 학습했어요. 예: ${wkLabel(WK[WK.length - 1])} 주는 ~${md(WK[WK.length - 1].train_until)} 데이터로 학습.</li>
        <li>모델 구조와 설정은 10번(M3) 본 모델과 같고, 트리 수만 학습 데이터 양에 비례해 맞췄어요. 9/1 주 결과는 기존 test 결과(8/31까지 학습)와 완전히 같은지 확인했어요.</li>
        <li>당일 피크 경보(M2)는 다시 학습하면 몇 시간이 걸려, 이미 있는 <b>월 단위</b> 결과(그 달 이전 데이터로 학습)를 썼어요.</li>
        <li>1~6월은 다른 날짜를 복사한 '복제일'이 많아 실제와 비교가 왜곡돼서 7월부터 봤어요.</li></ul>`) +
      acc('AI 계획 · 사후 최선 · 실제는 어떻게 다른가요?', `<ul>
        <li><b>실제 운영</b>: 그 주의 실제 전력이에요.</li>
        <li><b>AI 계획</b>: 미래 정보 없이 하루 전 예측과 평소 일정 계획으로 조정안을 짜고, 같은 이동량을 실제 전력에 적용한 결과예요. AI를 실제로 썼을 때의 효과예요.</li>
        <li><b>사후 최선</b>: 같은 조정 규칙에 예측 대신 <b>실제 전력</b>(완벽한 예측)과 실제 가동 기록을 넣은 결과예요. 이 규칙으로 도달할 수 있는 상한이에요.</li>
        <li>AI 계획과 사후 최선의 차이가 <b>예측 오차로 놓친 몫</b>이고, 모델이 나아질수록 줄어들어야 해요.</li></ul>`) : '';
    $('#a-method').innerHTML = '<p class="muted">하루 전 예측으로 일정을 짜고, 같은 이동량을 그날의 실제 전력에 적용해 효과를 계산했습니다.</p>' + fbAcc +
      acc('어떻게 옮기나요?', `<ul>
        <li><b>덜어내는 시간</b>: 계획상 가동 중이고 (예측 + 안전 마진)이 ${fmt(T - 1, 0)}kW를 넘는 시간. 넘는 만큼만, 최대 이동 가능 비율까지.</li>
        <li><b>받는 시간</b>: 가동 중이고 덜어내지 않는 시간 중 예측이 낮은 곳부터. 받은 뒤에도 (예측 + 마진)이 ${fmt(T - 1 - D.meta.dest_buffer_kw, 0)}kW 이하가 되게 합니다. 같은 날, 최대 ±${D.meta.max_shift_h}시간 안에서만 옮깁니다.</li></ul>`) +
      acc('안전 마진은 어떻게 정했나요?', `<p>M3는 일 최대전력을 낮게 예측합니다(편향: 7월 ${b['2021-07']}, 8월 ${b['2021-08']}, test ${b.test}kW). 기본 ${D.meta.margins[1]}kW는 CV 두 달의 편향에서 정했고, test 결과를 보고 고르지 않았습니다.</p>`) +
      acc('예측은 얼마나 정확한가요?', `<ul><li>M3 test MAE ${m.m3_mae}kW, 일 최대 MAE ${m.m3_daily_max_mae}kW</li>
        <li>M2 주의 단계 정밀도 ${m.m2_alert.precision}, 재현율 ${m.m2_alert.recall}</li></ul>`) +
      acc('기본요금은 어떻게 계산하나요?', `<p>한국전력은 15분 단위 최대수요전력 중 가장 높은 값을 기준으로 요금을 부과합니다(가이드북). 절감액 = 최대수요 절감(kW) × 단가 × 12개월. 기본 단가 ${TARIFF.price.toLocaleString('ko-KR')}${TARIFF.unit}은 ${TARIFF.label.replace(' (추정)', '')}로 추정한 값이라(${TARIFF.basis.replace(' · ', ', ')}, 데이터에 계약 종별이 없음), 실제 계약이 다르면 직접 바꿔 입력하세요.</p>`) +
      acc('반영하지 못한 것', '<p>작업별 이동 가능 여부, 점심·교대 시간, 인원 배치는 데이터에 없어 반영하지 않았습니다. 낮 부하를 새벽·야간으로 옮기면 야간 인건비(데이터의 인건비 비율 1.5)가 늘 수 있는데 이 비용도 계산에 넣지 않았습니다. 받는 시간이 실제로 작업 가능한지는 담당자가 확인해야 합니다.</p>') +
      acc('시뮬레이션 주의', `<p>${D.meta.notes.join(' ')}</p>`);
  }

  // ── 주간 피드백 (15_weekly_feedback.py: 주마다 전주까지 학습 → AI 계획 vs 사후 최선 vs 실제) ──
  const WK = D.weeks || null;
  const wkOf = (d) => (WK ? Math.max(0, WK.findIndex((w) => w.days.includes(d))) : 0);
  const wk = () => WK[S.aweek];
  const wkSc = (i = S.aweek) => WK[i].scenarios[keyOf()];
  const hsOf = (d) => (D.hindsight ? (D.hindsight[`r${Math.round(S.share * 100)}`] || {})[d] : null);
  const isExcluded = (d) => (D.meta.excluded_days || []).includes(d);
  const wkLabel = (w) => `${md(w.start)} ~ ${md(w.end)}`;
  const weekDays = () => (WK ? wk().days.filter((d) => D.days[d]) : dates);

  function initWeeks() {
    if (!WK) return;
    $('#a-week-card').hidden = false;
    $('#a-trend-card').hidden = false;
    $('#a-week').innerHTML = WK.map((w, i) => `<option value="${i}">${wkLabel(w)} · 학습 ~${md(w.train_until)}</option>`).join('');
    const go = (i) => {
      S.aweek = Math.max(0, Math.min(WK.length - 1, i));
      const ds = weekDays();
      S.date = ds.reduce((a, b) => (peakHoursOf(b) > peakHoursOf(a) ? b : a), ds[0]);
      renderAdmin(); live(`${wkLabel(wk())} 주를 봅니다`);
    };
    $('#a-week').onchange = (e) => go(+e.target.value);
    $('#a-prev').onclick = () => go(S.aweek - 1);
    $('#a-next').onclick = () => go(S.aweek + 1);
    $('#a-trend-table-toggle').onclick = (e) => {
      const open = $('#a-trend-table').hidden;
      $('#a-trend-table').hidden = !open;
      e.target.setAttribute('aria-expanded', String(open));
      e.target.textContent = open ? '표 닫기' : '표로 보기';
    };
    S.goWeek = go;
  }

  function renderWeekHead() {
    const w = wk(), prev = WK[S.aweek - 1];
    $('#a-week').value = String(S.aweek);
    $('#a-prev').disabled = S.aweek === 0;
    $('#a-next').disabled = S.aweek === WK.length - 1;
    const ex = w.excluded_days.length ? ` ${w.excluded_days.map(md).join(', ')}은 다른 날짜를 복사한 '복제일'이라 지표에서 뺐어요.` : '';
    // 한 줄 결론을 맨 위에, 학습 조건 설명은 접어 둔다
    const a = wkSc().ai, h = wkSc().hs;
    const pct = a.peak_hours_actual ? Math.round((1 - a.peak_hours / a.peak_hours_actual) * 100) : 0;
    $('#a-headline').innerHTML = a.peak_hours_actual
      ? `AI 계획으로 피크 시간이 <span class="hl">${pct}% 줄었어요</span> <span class="sub">${a.peak_hours_actual} → ${a.peak_hours}시간 · 예측이 완벽했다면 ${h.peak_hours}시간</span>`
      : '이번 주는 피크 시간이 없었어요';
    $('#a-week-lead').innerHTML = acc(`${md(w.train_until)}까지 학습한 모델로 예측 · 어떻게 계산했나요?`,
      `<p>${wkLabel(w)}을 ${md(w.train_until)}까지의 데이터(${w.train_rows.toLocaleString('ko-KR')}시간)로 다시 학습한 모델로 하루 전에 예측하고, 일정 조정안을 실제 전력에 적용해 봤어요.${prev ? ` 지난주보다 학습 데이터가 ${(w.train_rows - prev.train_rows).toLocaleString('ko-KR')}시간 늘었어요.` : ''}${ex}</p>`);
  }

  function renderWeekKpis(animate = true) {
    const s = wkSc(), a = s.ai, h = s.hs;
    const cutAi = a.week_max_actual - a.week_max, cutHs = a.week_max_actual - h.week_max;
    // 핵심 3개만: 요금 기준 최대수요 · 피크 시간 · 월 절감액 (예측 오차는 아래 '주별 추이'에서)
    $('#kpis').innerHTML = `
      <div class="kpi"><div class="label">피크 시간 <span class="unit">${T}kW 이상</span></div>
        <div class="value"><span class="cu" data-from="${a.peak_hours_actual}" data-to="${a.peak_hours}" data-dec="0">${a.peak_hours}</span><span class="u">시간</span></div>
        <div class="was">실제 ${a.peak_hours_actual}시간에서 ${badge(a.peak_hours_actual - a.peak_hours, '시간')}</div>
        <div class="foot">예측이 완벽했다면 ${h.peak_hours}시간</div></div>
      <div class="kpi"><div class="label">주간 최대수요 <span class="unit">요금 기준</span></div>
        <div class="value"><span class="cu" data-from="${a.week_max_actual}" data-to="${a.week_max}" data-dec="1">${fmt(a.week_max)}</span><span class="u">kW</span></div>
        <div class="was">실제 ${fmt(a.week_max_actual)}kW에서 ${badge(cutAi, 'kW', 1)}</div>
        <div class="foot">예측이 완벽했다면 ${fmt(h.week_max)}kW</div></div>
      <div class="kpi"><div class="label">기본요금 절감 ${tariffTag()}</div>
        <div class="value"><span class="cu" data-from="0" data-to="${Math.round(Math.max(0, cutAi) * S.price)}" data-dec="0">${fmt(Math.round(Math.max(0, cutAi) * S.price), 0)}</span><span class="u">원/월</span></div>
        <div class="was">예측이 완벽했다면 월 ${won(Math.max(0, cutHs) * S.price)}</div>
        <div class="foot" title="${tariffFoot('최대수요 절감')}">12개월 유지 시 추정</div></div>`;
    if (animate) countUp($('#kpis'));
  }
  // 줄어든 양 표시: 줄면 파랑 '−n', 늘면 주황 '+n', 그대로면 회색
  const badge = (cut, unit, dec = 0) => (Math.abs(cut) < (dec ? 0.05 : 0.5)
    ? '<span class="badge same">변화 없음</span>'
    : `<span class="badge ${cut > 0 ? 'down' : 'up'}">${cut > 0 ? '−' : '+'}${fmt(Math.abs(cut), dec)}${unit}</span>`);

  function renderWeekCallout() {
    const w = wk();
    const notes = w.notes.length ? `<ul>${w.notes.map((n) => `<li>${n}</li>`).join('')}</ul>` : '<p>이번 주는 특별히 반영할 점이 없어요.</p>';
    $('#callout').innerHTML = `<p class="lead"><strong>다음 주에 반영할 점</strong></p>${notes}`;
  }

  function trendChart(el, series, ytitle, fmtv) {
    const n = WK.length, W = 480, H = 220, L = 40, R = 12, Tp = 14, B = 34, iw = W - L - R, ih = H - Tp - B;
    const vals = series.flatMap((s) => s.v);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const pad0 = (hi - lo) * 0.15 || 5; lo = Math.max(0, Math.floor((lo - pad0) / 5) * 5); hi = Math.ceil((hi + pad0) / 5) * 5;
    const x = (i) => L + (iw * i) / (n - 1), y = (v) => Tp + ih * (1 - (v - lo) / (hi - lo));
    const step = (hi - lo) / 4;
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${ytitle} 주별 추이. 표로 보기에서 값을 볼 수 있습니다.">`;
    for (let k = 0; k <= 4; k++) { const v = lo + step * k; s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${Math.round(v)}</text>`; }
    s += `<rect x="${x(S.aweek) - iw / (n - 1) / 2}" y="${Tp}" width="${iw / (n - 1)}" height="${ih}" style="fill:var(--in-bg);opacity:.7"/>`;
    WK.forEach((w, i) => { if (i % 2 === 0 || i === n - 1) s += `<text x="${x(i)}" y="${H - 14}" text-anchor="middle">${md0(w.start)}</text>`; });
    if (series.some((sr) => sr.ref)) s += `<line x1="${L}" x2="${W - R}" y1="${y(T)}" y2="${y(T)}" style="stroke:var(--text-3)" stroke-width="1" stroke-dasharray="5 4"/>`;
    series.forEach((sr) => {
      s += `<path d="${sr.v.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('')}" fill="none" style="stroke:${sr.color}" stroke-width="2"${sr.dash ? ' stroke-dasharray="5 4"' : ''}/>`;
      sr.v.forEach((v, i) => { s += `<circle cx="${x(i)}" cy="${y(v)}" r="${i === S.aweek ? 5 : 3.5}" style="fill:${sr.color};stroke:var(--surface)" stroke-width="2"/>`; });
    });
    WK.forEach((w, i) => { s += `<rect class="pt" data-i="${i}" x="${x(i) - iw / (n - 1) / 2}" y="${Tp}" width="${iw / (n - 1)}" height="${ih}" fill="transparent"><title>${wkLabel(w)}: ${series.map((sr) => `${sr.name} ${fmtv(sr.v[i])}`).join(', ')}</title></rect>`; });
    s += '</svg>';
    el.innerHTML = s;
    el.querySelectorAll('.pt').forEach((r) => { r.onclick = () => S.goWeek(+r.dataset.i); });
  }

  function renderTrend() {
    const k = keyOf();
    const act = WK.map((w) => w.scenarios[k].ai.week_max_actual), ai = WK.map((w) => w.scenarios[k].ai.week_max), hs = WK.map((w) => w.scenarios[k].hs.week_max);
    $('#a-trend-legend').innerHTML = `
      <span><i class="sw" style="background:var(--orange)"></i>실제 운영</span>
      <span><i class="sw" style="background:var(--blue)"></i>AI 계획 (전주까지 학습)</span>
      <span><i class="sw" style="background:var(--text-3)"></i>사후 최선 (점선, 예측이 완벽했다면)</span>
      <span><i class="band" style="background:var(--in-bg);border:1px solid var(--line)"></i>선택한 주</span>`;
    trendChart($('#a-trend-max'), [
      { name: '실제', v: act, color: 'var(--orange)', ref: true },
      { name: 'AI 계획', v: ai, color: 'var(--blue)' },
      { name: '사후 최선', v: hs, color: 'var(--text-3)', dash: true },
    ], '주간 최대수요', (v) => fmt(v) + 'kW');
    const w = wk(), prev = WK[S.aweek - 1], dMae = prev ? w.m3.mae - prev.m3.mae : null;
    $('#a-trend-mae-title').innerHTML = `하루 전 예측 오차 MAE (kW) · 이번 주 <b>${fmt(w.m3.mae)}</b>${dMae === null ? '' : ` (지난주 대비 ${dMae <= 0 ? '−' : '+'}${fmt(Math.abs(dMae))})`}`;
    trendChart($('#a-trend-mae'), [{ name: 'MAE', v: WK.map((w) => w.m3.mae), color: 'var(--blue)' }], '하루 전 예측 오차', (v) => fmt(v, 2) + 'kW');
    $('#a-trend-table').innerHTML = `<table><thead><tr><th>주</th><th>학습 데이터</th><th class="num">MAE</th><th class="num">최대수요 실제 / AI / 사후</th><th class="num">피크 시간 실제 / AI / 사후</th><th class="num">M2 경보</th></tr></thead><tbody>${WK.map((w, i) => {
      const s = w.scenarios[k];
      return `<tr${i === S.aweek ? ' class="sel"' : ''}><td>${wkLabel(w)}</td><td>~${md(w.train_until)}</td><td class="num">${fmt(w.m3.mae, 2)}</td><td class="num">${fmt(s.ai.week_max_actual, 0)} / ${fmt(s.ai.week_max, 0)} / ${fmt(s.hs.week_max, 0)}</td><td class="num">${s.ai.peak_hours_actual} / ${s.ai.peak_hours} / ${s.hs.peak_hours}</td><td class="num">${w.m2.caught}/${w.m2.peaks}</td></tr>`;
    }).join('')}</tbody></table>`;
  }

  function renderWeekDays() {
    $('#a-days-title').textContent = '이번 주 날짜별 (실제 → AI 계획 → 사후 최선)';
    const rows = weekDays().map((d) => {
      const st = stats(d), hs = hsOf(d);
      const hMax = hs ? maxOf(hs.actual_after) : null, hPeak = hs ? hs.actual_after.filter((v) => v >= T).length : null;
      return `<tr class="click${d === S.date ? ' sel' : ''}" data-d="${d}" tabindex="0">
        <td>${md(d)}${isExcluded(d) ? ' <span class="tag">복제일 · 지표 제외</span>' : ''}</td><td>${st.raw.day_type}</td>
        <td class="num">${fmt(st.bMax, 0)} → ${fmt(st.aMax, 0)} → ${hMax === null ? '-' : fmt(hMax, 0)}</td>
        <td class="num">${st.bPeak} → ${st.aPeak} → ${hPeak === null ? '-' : hPeak}</td><td class="num">${fmt(st.moved, 0)}</td></tr>`;
    }).join('');
    $('#a-days').innerHTML = `<table><thead><tr><th>날짜</th><th>날 유형(M1)</th><th class="num">일 최대 kW</th><th class="num">피크 시간</th><th class="num">AI 이동량 kW</th></tr></thead><tbody>${rows}</tbody></table>`;
    document.querySelectorAll('#a-days tr.click').forEach((tr) => {
      const go = () => { S.date = tr.dataset.d; renderAdmin(); $('#a-chart').scrollIntoView({ behavior: 'smooth', block: 'center' }); };
      tr.onclick = go;
      tr.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } };
    });
  }

  // 날짜 고르기: 드롭다운 대신 그 주 날짜를 버튼으로 펼쳐 피크가 있던 날이 바로 보이게 한다
  function renderDayPick() {
    const ds = weekDays();
    $('#a-days-pick').innerHTML = ds.map((d) => {
      const p = peakHoursOf(d);
      return `<button type="button" class="day-chip${p ? ' has-peak' : ''}" data-d="${d}" aria-pressed="${d === S.date}">
        <span class="dd">${md(d)}</span><span class="pk">${p ? `피크 ${p}시간` : '피크 없음'}</span></button>`;
    }).join('');
    $('#a-days-pick').querySelectorAll('.day-chip').forEach((b) => {
      b.onclick = () => { S.date = b.dataset.d; renderAdmin(); };
    });
    showPressed($('#a-days-pick'));
  }

  function renderAdmin() {
    if (WK && !weekDays().includes(S.date)) S.aweek = wkOf(S.date);   // 다른 화면에서 날짜를 바꿨으면 그 주로
    renderDayPick();
    if (WK) { renderWeekHead(); renderWeekKpis(); renderWeekCallout(); renderTrend(); renderChart(); renderChanges(); renderWeekDays(); renderMethod(); return; }
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
  const hrs = (s, e) => `${pad(s)}–${pad(e + 1)}시`;
  const outRuns = (d) => (D.days[d] ? runs(d).filter((r) => r.st === 'out') : []);
  const wDays = (i = S.week) => Array.from({ length: 7 }, (_, k) => addDays(weeks[i], k));
  // 고른 블록과 짝이 되는 시간: 덜어내는 블록이면 받는 시간, 받는 블록이면 덜어내는 시간
  function pairHours(r) {
    const adj = adjOf(r.d), set = new Set();
    adj.transfers.forEach((t) => {
      if (r.st === 'out' && t.from >= r.s && t.from <= r.e) set.add(t.to);
      if (r.st === 'in' && t.to >= r.s && t.to <= r.e) set.add(t.from);
    });
    return set;
  }
  const runOf = (sel) => (sel && D.days[sel.d] ? runs(sel.d).find((r) => r.s === sel.s) : null);
  const biggest = (rs) => rs.reduce((a, r) => (!a || r.kw > a.kw ? r : a), null);

  function renderWorker() {
    const mon = weeks[S.week], days7 = wDays();
    // 선택이 없거나 다른 주면: 고른 날(없으면 그 주에서 가장 큰 작업이 있는 날)의 가장 큰 작업
    if (!days7.includes(S.date) || !D.days[S.date]) {
      const best = biggest(days7.flatMap(outRuns));
      S.date = best ? best.d : days7.find((d) => D.days[d]) || S.date;
    }
    const folded = S.detail && S.detail.s === -1 && S.detail.d === S.date;   // 할 일 목록에서 접은 상태
    if (S.detail && !folded && (S.detail.d !== S.date || !runOf(S.detail))) S.detail = null;
    if (!S.detail) { const b = biggest(outRuns(S.date)); S.detail = b ? { d: b.d, s: b.s, e: b.e } : null; }
    const cur = runOf(S.detail), pairs = cur ? pairHours(cur) : new Set();

    $('#w-range').textContent = `${md0(mon)} ~ ${md0(addDays(mon, 6))}`;
    $('#w-prev').disabled = S.week === 0;
    $('#w-next').disabled = S.week === weeks.length - 1;
    const all = days7.flatMap(outRuns), top = biggest(all);
    $('#w-headline').innerHTML = all.length
      ? `이번 주 옮길 작업 <span class="hl">${all.length}건</span><span class="sub">가장 큰 작업: ${md(top.d)} ${hrs(top.s, top.e)} ${shiftTxt(top.kw, '−')}</span>`
      : '이번 주는 옮길 작업이 없어요<span class="sub">평소 일정대로 가동하면 돼요.</span>';
    $('#w-days').innerHTML = days7.map((d) => {
      const has = !!D.days[d], n = outRuns(d).length;
      return `<button type="button" class="day-chip${n ? ' has-peak' : ''}" data-d="${d}" aria-pressed="${d === S.date}"${has ? '' : ' disabled'}>
        <span class="dd">${md(d)}</span><span class="pk">${!has ? '자료 없음' : n ? `옮길 작업 ${n}건` : '평소대로'}</span></button>`;
    }).join('');
    $('#w-days').querySelectorAll('.day-chip').forEach((b) => { b.onclick = () => { S.date = b.dataset.d; S.detail = null; renderWorker(); }; });
    showPressed($('#w-days'));

    $('#w-legend').innerHTML = ['out', 'in', 'normal', 'off'].map((k) => `<span><i class="lg-${k}"></i>${NAME[k]}</span>`).join('');
    let html = '<div class="hd"></div>';
    days7.forEach((d, i) => {
      const has = !!D.days[d];
      html += `<div class="hd${d === S.date ? ' sel' : ''}${has ? '' : ' dim'}"${has ? ` data-d="${d}" role="button" tabindex="0"` : ''}>${DOW[i]}<small>${md0(d)}</small></div>`;
    });
    html += '<div class="hours">' + Array.from({ length: 24 }, (_, h) => `<div>${pad(h)}</div>`).join('') + '</div>';
    days7.forEach((d, ci) => {
      if (!D.days[d]) { html += '<div class="col nodata">자료 없음</div>'; return; }
      html += `<div class="col${d === S.date ? ' today' : ''}">` + runs(d).map((r, bi) => {
        const len = r.e - r.s + 1;
        const sel = cur && cur.d === r.d && cur.s === r.s;
        const pair = cur && cur.d === r.d && !sel && [...pairs].some((h) => h >= r.s && h <= r.e);
        const sub = r.st === 'out' ? shiftTxt(r.kw, '−') : r.st === 'in' ? shiftTxt(r.kw, '+') : '';
        // 일반 가동·비가동은 이름을 짧은 블록에서 생략해 화면을 덜 복잡하게
        const quiet = (r.st === 'normal' || r.st === 'off') && len < 3;
        return `<button type="button" class="blk ${r.st}${pair ? ' pair' : ''}" aria-pressed="${sel ? 'true' : 'false'}" data-d="${r.d}" data-s="${r.s}" data-e="${r.e}"
          style="--ci:${ci};--bi:${bi};top:calc(var(--rowh)*${r.s} + 1px);height:calc(var(--rowh)*${len} - 3px)"
          aria-label="${md(r.d)} ${hh(r.s)}부터 ${hh(r.e + 1)}까지 ${NAME[r.st]} ${sub}">
          ${quiet ? '' : len === 1   // 1시간 블록은 한 줄만 들어가서 짧은 이름으로
            ? `<span class="nm one">${{ out: '자제', in: '배치' }[r.st] || NAME[r.st]}${sub ? ' ' + sub : ''}</span>`
            : `<span class="nm">${NAME[r.st]}${sub ? ' ' + sub : ''}</span><span class="tm">${hrs(r.s, r.e)}</span>`}</button>`;
      }).join('') + '</div>';
    });
    const tt = $('#tt');
    tt.innerHTML = html;
    tt.classList.toggle('focus', !!cur && (cur.st === 'out' || cur.st === 'in'));
    tt.querySelectorAll('.blk').forEach((b) => {
      b.onclick = () => { S.detail = { d: b.dataset.d, s: +b.dataset.s, e: +b.dataset.e }; S.date = b.dataset.d; renderWorker(); };
    });
    tt.querySelectorAll('.hd[data-d]').forEach((h) => {
      const go = () => { S.date = h.dataset.d; S.detail = null; renderWorker(); };
      h.onclick = go;
      h.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } };
    });
    renderWorkerDetail(cur);
  }
  const md0 = (d) => `${+d.slice(5, 7)}/${+d.slice(8)}`;

  function blockText(r) {
    const raw = D.days[r.d], adj = adjOf(r.d), hs = [];
    for (let h = r.s; h <= r.e; h++) hs.push(h);
    const peak = Math.max.apply(null, hs.map((h) => raw.pred[h]));
    if (r.st === 'out') {
      const to = [...pairHours(r)];
      return { what: '큰 설비 가동·예열 미루기', to: to.length ? `→ ${rangeText(to)}로 옮기기` : '',
        more: `<p>설비가 한꺼번에 돌아 전력이 몰릴 것으로 예상돼요 (예상 최대 <b>${fmt(peak, 0)}kW</b>, 기준 ${fmt(T, 0)}kW).</p>
          <p>미룰 수 있는 작업을 옮겨 주세요. 총 ${fmt(hs.reduce((a, h) => a + adj.out[h], 0), 0)}kW만큼 덜어내는 것이 목표예요.</p>` };
    }
    if (r.st === 'in') {
      const from = [...pairHours(r)];
      return { what: '미룬 작업 몰아서 하기', to: from.length ? `← ${rangeText(from)}에서 가져오기` : '',
        more: `<p>예상 <b>${fmt(peak, 0)}kW</b>로 여유가 있어요. 최대 ${fmt(hs.reduce((a, h) => a + adj.in[h], 0), 0)}kW까지 받을 수 있어요.</p>` };
    }
    if (r.st === 'off') return { what: '계획상 가동하지 않는 시간', to: '', more: '' };
    return { what: '평소대로 가동', to: '', more: `<p>예상 최대 ${fmt(peak, 0)}kW로 조정할 필요가 없어요.</p>` };
  }

  // 오른쪽(휴대폰은 위쪽) 패널: 고른 날 할 일 목록. 고른 작업은 펼쳐서 이유를 보여 준다
  function renderWorkerDetail(cur) {
    const box = $('#w-detail'), d = S.date;
    if (!D.days[d]) { box.innerHTML = '<p class="empty">이 주에는 자료가 없어요.</p>'; return; }
    const items = runs(d).filter((r) => r.st === 'out' || r.st === 'in');
    const sel = (r) => cur && cur.s === r.s;
    const item = (r) => {
      const t = blockText(r), amt = r.st === 'out' ? shiftTxt(r.kw, '−') : shiftTxt(r.kw, '+');
      return `<li class="todo ${r.st}${sel(r) ? ' sel' : ''}"><button type="button" data-s="${r.s}" data-e="${r.e}" aria-expanded="${sel(r)}">
          <span class="tm">${hrs(r.s, r.e)}</span>
          <span class="what">${t.what}<small>${t.to}</small></span>
          <span class="amt">${amt}</span></button>
          ${sel(r) && t.more ? `<div class="todo-more">${t.more}</div>` : ''}</li>`;
    };
    const other = cur && cur.st !== 'out' && cur.st !== 'in' ? blockText(cur) : null;
    box.innerHTML = `<h2>${md(d)} 할 일</h2>
      <p class="muted">${D.days[d].day_type} · ${items.length ? `옮길 작업 ${items.filter((r) => r.st === 'out').length}건` : '평소대로 가동하면 돼요'}</p>
      ${other ? `<div class="todo-other"><b>${hrs(cur.s, cur.e)} · ${other.what}</b>${other.more}</div>` : ''}
      ${items.length ? `<ol class="todos">${items.map(item).join('')}</ol>` : '<p class="empty">이 날은 전력이 몰리는 시간이 없어요.</p>'}
      <p class="small">관리자 화면과 같은 계산 결과예요. 실제 작업 가능 여부는 현장 상황에 맞게 조정하세요.</p>`;
    box.querySelectorAll('.todo > button').forEach((b) => {
      b.onclick = () => {
        const same = cur && cur.s === +b.dataset.s;
        S.detail = same ? { d, s: -1, e: -1 } : { d, s: +b.dataset.s, e: +b.dataset.e };   // 다시 누르면 접기
        renderWorker();
      };
    });
  }

  // ── 챗봇 ─────────────────────────────────────────────
  const log = $('#chat-log');
  const BOT_ICON = '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><path d="M13 2 4 14h7l-1 8 9-12h-7l1-8Z" fill="currentColor"/></svg>';
  function addMsg(who, html, acts) {
    $('#chat-empty').hidden = true;   // 첫 질문부터는 빈 화면 안내 대신 대화와 아래 칩을 보여 준다
    $('#chips').hidden = false;
    const row = document.createElement('div');
    row.className = 'msg-row ' + who;
    if (who === 'bot') row.innerHTML = `<span class="avatar sm">${BOT_ICON}</span>`;
    const el = document.createElement('div');
    el.className = 'msg ' + who;
    if (who === 'user') el.textContent = html; else el.innerHTML = html;
    if (acts && acts.length) {
      const a = document.createElement('div'); a.className = 'acts';
      acts.forEach((x) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = x.label; b.onclick = x.fn; a.appendChild(b); });
      el.appendChild(a);
    }
    row.appendChild(el);
    log.appendChild(row);
    log.scrollTop = log.scrollHeight;
    updateCtx();
  }
  const updateCtx = () => { $('#chat-ctx').innerHTML = `기준 날짜 <b>${md(S.date)}</b>`; };
  const goto = (view, label) => ({ label, fn: () => setView(view) });

  function rawDate(t) {
    let m = /(\d{1,2})\s*(?:\/|월|\.)\s*(\d{1,2})/.exec(t);
    let d = m ? `2021-${pad(+m[1])}-${pad(+m[2])}` : null;
    if (!d) { m = /(\d{1,2})\s*일/.exec(t); d = m ? `2021-09-${pad(+m[1])}` : null; }
    return d;
  }
  function parseDate(t) {
    const d = rawDate(t);
    if (!d) return null;
    if (D.days[d]) return d;
    return PA && planKeyFor(d) ? d : null;   // 9/15 이후 등 주간 계획 결과에만 있는 날짜
  }
  const isPlanOnly = (d) => !D.days[d];

  // 주간 계획 답변 (13·14 결과)
  const riskHours = (w, d, key) => w.time.map((t, i) => (t.startsWith(d) && w.hours[key][i] + PA.model.margin >= T ? +t.slice(11, 13) : null)).filter((h) => h !== null);
  function moveLine(m) {
    if (m.kind === 'day') {
      return `${dlabel(m.from)} 물량 ${num0(m.qty)} → ${dlabel(m.to)}${m.new_hours.length ? ` (${dlabel(m.to)} ${m.new_hours.map((h) => h.slice(11, 13) + '시').join(', ')} 새로 가동, 실험적)` : ''}`;
    }
    return `${tlabel(m.from)} 물량 ${num0(m.qty)} → 같은 날 ${m.to.slice(11, 13)}시`;
  }
  function ansPlanWeek(k) {
    const w = PW[k], s = w.summary, days = w.moves.filter((m) => m.kind === 'day');
    const cut = s.pred_week_max_before - s.pred_week_max_after;
    let html = `<p><b>${w.label}</b> (${dlabel(s.start)}~${dlabel(s.end)}, ${w.ext ? '적극안' : '기본안'}) 수정안이에요.</p>
      <p>주간 예측 최대 ${fmt(s.pred_week_max_before)} → <b>${fmt(s.pred_week_max_after)}kW</b>${cut > 0.05 ? ` (−${fmt(cut)}kW)` : ''}, 일 최대가 내려간 날 ${s.days_lowered}일 평균 −${fmt(s.avg_cut_on_lowered_days)}kW예요.</p>`;
    if (s.backtest) html += `<p>실제 피크 시간은 ${s.actual_peak_hours} → 약 ${s.actual_peak_hours_after_est}시간(추정)으로 줄어요.</p>`;
    html += w.moves.length
      ? acc(`바꾸는 내용 ${w.moves.length}건 보기`, `<ul>${(days.length ? days : w.moves).map((m) => `<li>${moveLine(m)}</li>`).join('')}</ul><p>주간 총 생산량은 그대로예요.</p>`)
      : '<p>규칙 안에서 최대전력을 의미 있게 낮추는 이동은 찾지 못했어요.</p>';
    return html;
  }
  function ansPlanDay(k, d) {
    const w = PW[k], r = w.days.find((x) => x.date === d);
    if (!r) return ansPlanWeek(k);
    const rb = riskHours(w, d, 'pred_before'), ra = riskHours(w, d, 'pred_after');
    const mv = w.moves.filter((m) => m.from.startsWith(d) || m.to.startsWith(d));
    let html = `<p><b>${dlabel(d)}</b> · ${w.label} (${w.ext ? '적극안' : '기본안'}) 기준이에요.</p>`;
    html += r.on_before === 0 && r.on_after === 0
      ? '<p>계획상 가동하지 않는 날이에요.</p>'
      : `<p>예측 최대 ${fmt(r.pred_max_before, 0)} → <b>${fmt(r.pred_max_after, 0)}kW</b>, 위험 시간 ${rb.length ? rangeText(rb) : '없음'}${ra.length !== rb.length ? ` → 수정 후 ${ra.length ? rangeText(ra) : '없음'}` : ''}</p>`;
    html += mv.length
      ? `<p>이 날 바꾸는 것:</p><ul>${mv.map((m) => `<li>${moveLine(m)}</li>`).join('')}</ul>`
      : '<p>이 날은 바꾸는 물량이 없어요.</p>';
    if (r.out_of_range) html += `<p><strong>주의:</strong> 이 날 계획 생산량(${num0(r.prod_before)})은 과거 하루 생산량의 99% 수준보다 많아서, 예측과 수정안을 믿기 어려워요.</p>`;
    if (r.actual_max !== undefined) html += `<p class="small">실제 최대 ${fmt(r.actual_max, 0)}kW → 수정 후 추정 ${fmt(r.actual_max_after_est, 0)}kW</p>`;
    return html + acc('읽을 때 주의', `<p>1주 전 예측(위험 시간 = 예측 + 마진 ${fmt(PA.model.margin, 0)}kW ≥ ${fmt(T, 0)}kW)으로 계산했어요. 생산량을 옮겨 줄이는 폭은 날마다 몇~10kW 수준이고, 남는 피크는 당일 경보로 대응해야 해요.</p>`);
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
        <li>${S.price === TARIFF.price ? `${TARIFF.label} 단가 ${S.price.toLocaleString('ko-KR')}원/kW·월(${TARIFF.basis.replace(' · ', ', ')})` : `입력한 단가 ${S.price.toLocaleString('ko-KR')}원/kW·월`}이면 월 <b>${won(cut * S.price)}</b>, 연 ${won(cut * S.price * 12)} (12개월 유지 시 추정)</li>
        <li>요금적용전력은 직전 12개월 중 최대수요전력이라, 한 주 피크를 낮췄다고 바로 줄지 않아요. 또 모델은 1시간 평균, 요금은 15분 평균 기준이라 근사치예요.</li></ul>
        ${WK ? `<p>주별로 보면 ${wkLabel(wk())} 주는 AI 계획 ${fmt(wkSc().ai.week_max)}kW, 사후 최선 ${fmt(wkSc().hs.week_max)}kW예요. 관리자 화면에서 주를 바꿔 볼 수 있어요.</p>` : '<p>최대수요 절감이 작은 이유는 가장 높았던 날을 하루 전 예측이 놓쳤기 때문이에요.</p>'}`);
  }
  const ansLimits = () => '<p>시뮬레이션이라 실제 운영 결과와 다를 수 있어요.</p>' + acc('꼭 알아 둘 점 자세히 보기', `<ul>
      <li>${md(dates[0])}~${md(dates[dates.length - 1])} 데이터로 되돌려 본 결과예요.${WK ? ' 각 주는 그 주 시작 전날까지의 데이터로 다시 학습한 모델로 예측했어요.' : ''}</li>
      <li>하루 전 예측(M3)은 일 최대를 평균 8kW쯤 낮게 봐서, 그만큼 안전 마진을 더했어요.</li>
      <li>이동 가능 비율과 요금 단가는 데이터에 없어서 입력값이에요.</li>
      <li>작업별 이동 가능 여부, 점심·교대 시간은 반영하지 못했어요. 현장 확인이 필요해요.</li></ul>`);
  const ansHelp = () => `<p>이런 걸 물어보세요.</p><ul><li>"9/2 피크 위험 시간 알려줘"</li><li>"9/2 일정 추천해줘"</li><li>"왜 옮기는 거야?"</li><li>"절감 효과는?"</li><li>"한계가 뭐야?"</li>${PA ? '<li>"다음 주 계획 수정안 알려줘"</li><li>"9/16 계획 어떻게 바꿔?"</li>' : ''}</ul>`;

  function answer(text) {
    const t = text.replace(/\s+/g, '');
    const d = parseDate(text);
    const asked = rawDate(text);
    if (asked && !d) {   // 결과가 없는 날짜는 다른 날로 바꿔 답하지 않고 범위를 알려 준다
      const ranges = [`일정 조정(M4) ${md(dates[0])}~${md(dates[dates.length - 1])}`];
      if (PA) {
        const ks = Object.keys(PW).map((k) => PW[k].summary.start).concat(Object.keys(PW).map((k) => PW[k].summary.end)).sort();
        ranges.push(`주간 계획 ${dlabel(ks[0])}~${dlabel(ks[ks.length - 1])}`);
      }
      addMsg('bot', `<p>${+asked.slice(5, 7)}/${+asked.slice(8)} 계산 결과는 없어요. 볼 수 있는 기간: ${ranges.join(', ')}.</p>`, []);
      return;
    }
    const planOnly = d && isPlanOnly(d);
    if (d && !planOnly) { S.date = d; }
    const date = S.date;
    const acts = [goto('admin', '관리자 화면에서 보기'), goto('worker', '근로자 시간표 보기')];
    const planQ = /주간|계획|수정안|다음주|생산량|물량|1주/.test(t);
    let html, a = acts;
    if (/안녕|도움|help|뭘할|무엇/i.test(t)) html = ansHelp(), a = [];
    else if (/한계|가정|신뢰|정확|믿|조심|주의할/.test(t) && !planOnly) html = ansLimits(), a = [goto('admin', '계산 방법 보기')];
    else if (PA && (planOnly || planQ)) {
      // 9/15 이후 날짜이거나 주간 계획을 물으면 13·14 결과로 답한다
      let k = planKeyFor(d || null);
      if (!d && /다음주/.test(t) && k && PW[k].summary.backtest) {   // '다음 주'는 백테스트가 아닌 주로
        const g = planGroups().find((x) => !PW[variant(x, false) || variant(x, true)].summary.backtest);
        if (g) k = variant(g, false) || variant(g, true);
      }
      if (!k) html = '<p>그 날짜가 들어 있는 주간 계획 결과가 없어요.</p>', a = [];
      else {
        html = d ? ansPlanDay(k, d) : ansPlanWeek(k);
        a = [gotoPlan(k, '주간 계획 탭에서 보기')];
      }
      if (d) html = `<p class="small" style="margin:0 0 4px">${dlabel(d)} 기준으로 답할게요.</p>` + html;
      addMsg('bot', html, a);
      return;
    }
    else if (/왜|이유|근거/.test(t)) html = ansWhy(date);
    else if (/절감|효과|요약|얼마|요금|단가|최대수요|돈/.test(t)) html = ansSummary(), a = [goto('admin', '관리자 화면에서 보기')];
    else if (/피크|위험|경보|몰리|언제/.test(t)) html = ansPeak(date);
    else if (/일정|추천|시간표|스케줄|배치|작업|근로|뭐해|어떻게/.test(t)) html = ansSchedule(date);
    else html = `<p>그 질문은 아직 몰라요. 아래 질문 중에서 골라 보세요.</p>` + ansHelp(), a = [];
    if (d) html = `<p class="small" style="margin:0 0 4px">날짜를 ${md(d)}로 바꿨어요.</p>` + html;
    addMsg('bot', html, a);
  }

  // 빈 화면 추천 질문: 아이콘 + 질문 + 무엇을 알려 주는지
  const ICO = {
    peak: '<path d="M3 17l5-6 4 4 8-9" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
    cal: '<rect x="4" y="5" width="16" height="15" rx="3" fill="none" stroke="currentColor" stroke-width="2"/><path d="M4 10h16M9 3v4M15 3v4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
    why: '<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.3-1 .9-1 1.7M12 17h.01" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
    won: '<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><path d="M7.5 9l1.5 6 3-5 3 5 1.5-6M7 12h10" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>',
    plan: '<path d="M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>',
    warn: '<path d="M12 4 2.5 20h19L12 4Z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 10v4M12 17h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
  };
  function initChat() {
    const sugs = [
      ['peak', '피크 위험 시간 알려줘', '전력이 몰리는 시간과 경보'],
      ['cal', '일정 추천해줘', '옮길 작업과 받을 시간'],
      ['why', '왜 옮기는 거야?', '조정 이유와 근거'],
      ['won', '절감 효과는?', '기본요금 절감액 요약'],
    ].concat(PA ? [['plan', '다음 주 계획 수정안', '1주 전 예측으로 만든 수정안']] : [], [['warn', '한계가 뭐야?', '계산 가정과 주의할 점']]);
    $('#chat-sugs').innerHTML = sugs.map(([ic, q, sub]) => `<button type="button" class="sug" data-q="${q}">
      <span class="ic ${ic}"><svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">${ICO[ic]}</svg></span>
      <span><b>${q}</b><small>${sub}</small></span></button>`).join('');
    document.querySelectorAll('#chat-sugs .sug').forEach((b) => { b.onclick = () => ask(b.dataset.q); });
    const chips = ['피크 위험 시간 알려줘', '일정 추천해줘', '왜 옮기는 거야?', '절감 효과는?', '한계가 뭐야?'].concat(PA ? ['다음 주 계획 수정안', '9/17 계획 어떻게 바꿔?'] : []);
    $('#chips').innerHTML = chips.map((c) => `<button type="button">${c}</button>`).join('');
    document.querySelectorAll('#chips button').forEach((b) => { b.onclick = () => ask(b.textContent); });
    const inp = $('#chat-input'), send = $('#chat-send');
    inp.oninput = () => { send.disabled = !inp.value.trim(); };
    $('#chat-form').onsubmit = (e) => { e.preventDefault(); const v = inp.value.trim(); if (v) { inp.value = ''; send.disabled = true; ask(v); } };
    updateCtx();
  }
  // 답하기 전 잠깐 '입력 중' 점을 보여 준다 (동작 줄이기 설정이면 짧게)
  let busy = false;
  function ask(text) {
    if (busy) return;
    busy = true;
    addMsg('user', text);
    const typing = document.createElement('div');
    typing.className = 'msg-row bot typing';
    typing.innerHTML = `<span class="avatar sm">${BOT_ICON}</span><div class="msg bot"><i></i><i></i><i></i></div>`;
    log.appendChild(typing);
    log.scrollTop = log.scrollHeight;
    setTimeout(() => { typing.remove(); busy = false; answer(text); }, REDUCE ? 120 : 520);
  }

  // ── 주간 계획 (13 1주 전 예측 + 14 계획 수정 제안) ─────────
  const PA = D.plan_advice;
  const PW = PA ? PA.weeks : {};
  const groupOf = (k) => PW[k].group || k.replace(/_ext$/, '');
  const variant = (g, ext) => Object.keys(PW).find((k) => groupOf(k) === g && !!PW[k].ext === !!ext);
  // 선택한 주(group)에 고른 수정 범위가 없으면 있는 쪽을 보여 준다
  const pKey = () => variant(S.pweek, S.pext) || variant(S.pweek, !S.pext);
  const pw = () => (isEdit() ? ED.view : PW[pKey()]);
  // 내 계획(--publish)을 먼저, 그다음 예시
  const planGroups = () => {
    const ks = Object.keys(PW);
    const rank = (k) => (PW[k].source === 'user' ? 0 : PW[k].summary.backtest ? 2 : 1);  // 내 계획 → 다음 주 예시 → 백테스트
    const order = ks.slice().sort((a, b) => rank(a) - rank(b));
    return [...new Set(order.map(groupOf))];
  };
  const weekDates = (k) => { const w = PW[k]; return [w.summary.start, w.summary.end]; };
  const inWeek = (k, d) => { const [a, b] = weekDates(k); return d >= a && d <= b; };
  // 날짜가 들어 있는 주 결과: 지금 탭에서 고른 주 → 내 계획 → 예시 순서, 기본안 우선
  function planKeyFor(d) {
    const cur = pKey();
    if (cur && (!d || inWeek(cur, d))) return cur;
    for (const g of planGroups()) {
      const k = variant(g, false) || variant(g, true);
      if (!d || inWeek(k, d)) return k;
    }
    return null;
  }
  const dlabel = (d) => `${+d.slice(5, 7)}/${+d.slice(8, 10)}(${DOW[(new Date(d.slice(0, 10) + 'T00:00:00Z').getUTCDay() + 6) % 7]})`;
  const tlabel = (t) => `${dlabel(t)} ${t.slice(11, 13)}시`;
  const num0 = (v) => Math.round(v).toLocaleString('ko-KR');

  function initPlan() {
    if (!PA) return;
    const groups = planGroups();
    const edOpt = (e) => `<option value="${e.key}">✎ ${e.label} · ${dlabel(e.start)}~${dlabel(e.end)}</option>`;
    S.pweek = (S.shareEdit && EDITORS.some((e) => e.key === S.shareEdit.k)) ? S.shareEdit.k : (EDITORS[0] ? EDITORS[0].key : groups[0]);
    if (S.shareEdit) S.pext = !!S.shareEdit.x;
    $('#p-week').innerHTML = EDITORS.filter((e) => !e.backtest).map(edOpt).join('') + groups.map((g) => {
      const k = variant(g, false) || variant(g, true), w = PW[k];
      return `<option value="${g}">${w.label || g} · ${dlabel(w.summary.start)}~${dlabel(w.summary.end)}</option>`;
    }).join('') + EDITORS.filter((e) => e.backtest).map(edOpt).join('');
    initEditor();
    $('#p-week').onchange = (e) => { S.pweek = e.target.value; syncPlanPick(); renderPlan(); live('주를 바꿨습니다'); };
    $('#p-mode').onchange = (e) => {
      S.pext = e.target.value === '1';
      if (isEdit() && ED.res && ED.resExt !== S.pext) edStatus('수정 범위를 바꿨어요. "수정안 만들기"를 다시 누르세요.');
      renderPlan(); live('수정 범위를 바꿨습니다');
    };
    syncPlanPick();
    $('#p-table-toggle').onclick = (e) => {
      const open = $('#p-table').hidden;
      $('#p-table').hidden = !open;
      e.target.setAttribute('aria-expanded', String(open));
      e.target.textContent = open ? '표 닫기' : '표로 보기';
    };
  }

  // '수정 범위' 선택지는 고른 주에 실제로 있는 결과만 켠다
  function syncPlanPick() {
    if (isEdit()) {
      $('#p-mode').innerHTML = `<option value="0">기본안: 생산량 이동 (최대 ${Math.round(100 * PA.modes.base.max_move_share)}%)</option>
        <option value="1">적극안: 최대 ${Math.round(100 * PA.modes.ext.max_move_share)}% + 가동 시간대 연장 (실험적)</option>`;
      $('#p-week').value = S.pweek; $('#p-mode').value = S.pext ? '1' : '0';
      return;
    }
    const b = variant(S.pweek, false), e = variant(S.pweek, true);
    if (!(S.pext ? e : b)) S.pext = !b;
    const opt = (k, val, txt) => `<option value="${val}"${k ? '' : ' disabled'}>${txt}${k ? '' : ' (결과 없음)'}</option>`;
    const share = (k, def) => Math.round(100 * (k ? PW[k].max_move_share : def));
    $('#p-mode').innerHTML = opt(b, '0', `기본안: 생산량 이동 (최대 ${share(b, PA.modes.base.max_move_share)}%)`) +
      opt(e, '1', `적극안: 최대 ${share(e, PA.modes.ext.max_move_share)}% + 가동 시간대 연장 (실험적)`);
    $('#p-week').value = S.pweek;
    $('#p-mode').value = S.pext ? '1' : '0';
  }
  const gotoPlan = (k, label) => ({ label, fn: () => { S.pweek = groupOf(k); S.pext = !!PW[k].ext; syncPlanPick(); setView('plan'); } });

  // ── 계획 직접 편집 (web/plan_engine.js: 14_plan_advisor.py 와 같은 예측·수정안을 브라우저에서) ──
  const EDITORS = (PA && PA.editor) || [];
  const isEdit = () => !!S.pweek && EDITORS.some((e) => e.key === S.pweek);
  // ver: 계획이 바뀔 때마다 1씩 오른다. 수정안 계산 중에 계획이 바뀌었는지 확인하는 데 쓴다
  const ED = { engine: null, loading: null, key: null, on: null, prod: null, res: null, resExt: null, view: null, busy: false, paint: null, ver: 0 };

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const el = document.createElement('script');
      el.src = src; el.onload = resolve; el.onerror = () => reject(new Error(src + ' 를 불러오지 못했습니다'));
      document.head.appendChild(el);
    });
  }
  function loadEditor() {
    if (ED.engine) return Promise.resolve(ED.engine);
    if (!ED.loading) {
      ED.loading = (window.PlanEngine ? Promise.resolve() : loadScript('plan_engine.js'))
        .then(() => (window.PLAN_EDITOR_DATA ? null : loadScript('data/plan_editor.js')))
        .then(() => { ED.engine = window.PlanEngine.create(window.PLAN_EDITOR_DATA); return ED.engine; });
    }
    return ED.loading;
  }
  const edWeek = () => ED.engine.data.weeks[ED.key];
  const tpl = () => edWeek().template;
  const draftKey = () => `plan-draft-${ED.key}`;
  const edStatus = (msg) => { $('#pe-status').innerHTML = msg; };

  function openEditor(key) {
    ED.key = key; ED.res = null; ED.ver++;
    const t = tpl();
    ED.on = t.on.slice(); ED.prod = t.production.slice();
    if (S.shareEdit && S.shareEdit.k === key) {          // 공유 링크로 열었으면 그 계획
      applyDiff(S.shareEdit.c || []); S.shareEdit = null;
      edStatus('공유 링크의 계획을 불러왔어요.');
    } else {
      try {                                               // 이 브라우저에 남은 고친 계획
        const d = JSON.parse(localStorage.getItem(draftKey()) || 'null');
        if (d && d.length) { applyDiff(d); edStatus('이 브라우저에서 고치던 계획을 이어서 보여 줘요. 처음 계획으로 되돌리려면 "처음 계획으로"를 누르세요.'); }
        else edStatus('');
      } catch (e) { edStatus(''); }
    }
  }
  function diffNow() {
    const t = tpl(), out = [];
    for (let i = 0; i < t.on.length; i++) {
      const p = ED.on[i] ? Math.round(ED.prod[i] * 10) / 10 : 0, tp = t.on[i] ? t.production[i] : 0;
      if (ED.on[i] !== t.on[i] || Math.abs(p - tp) > 1e-9) out.push([i, ED.on[i], p]);
    }
    return out;
  }
  function applyDiff(c) {
    c.forEach(([i, o, p]) => { if (i >= 0 && i < ED.on.length) { ED.on[i] = o ? 1 : 0; ED.prod[i] = o ? Math.max(0, +p || 0) : 0; } });
  }
  function saveDraft() { try { localStorage.setItem(draftKey(), JSON.stringify(diffNow())); } catch (e) { /* 저장 불가 환경 */ } }

  // 현재 계획(+ 수정안)으로 화면용 결과를 만든다
  function edCompute() {
    const eng = ED.engine, prodNow = ED.prod.map((v, i) => (ED.on[i] ? v : 0));
    const pred = Array.from(eng.predict(ED.key, ED.on, prodNow));
    const valid = ED.res && ED.resExt === S.pext;
    const res = valid ? ED.res : { on: ED.on.slice(), production: prodNow, pred0: pred, pred1: pred, moves: [] };
    const w = eng.summarize(ED.key, ED.on, prodNow, res, { allowNew: S.pext });
    const e = EDITORS.find((x) => x.key === ED.key);
    ED.view = Object.assign(w, { label: e.label, source: 'edit', group: ED.key, max_move_share: PA.modes[S.pext ? 'ext' : 'base'].max_move_share, pending: !valid });
    $('#pe-apply').disabled = !valid || !ED.res.moves.length;
  }

  function renderGrid() {
    const w = edWeek(), t = tpl(), v = ED.view, thr = w.peak_threshold, n = ED.on.length;
    const pred = v.hours.pred_before;
    let hd = '<thead><tr><th class="hr">시각</th>';
    for (let d = 0; d < n / 24; d++) {
      const r = v.days[d];
      hd += `<th>${dlabel(w.time[d * 24])}<small>예측 최대 ${fmt(r.pred_max_before, 0)}</small></th>`;
    }
    hd += '</tr></thead>';
    let body = '<tbody>';
    for (let h = 0; h < 24; h++) {
      body += `<tr><th class="hr">${pad(h)}시</th>`;
      for (let d = 0; d < n / 24; d++) {
        const i = d * 24 + h, on = ED.on[i] === 1;
        const changed = ED.on[i] !== t.on[i] || (on && Math.abs(ED.prod[i] - t.production[i]) > 1e-9);
        const risk = pred[i] + w.margin >= thr;
        const cls = `pe-cell${on ? '' : ' off'}${changed ? ' changed' : ''}${risk ? ' risk' : ''}`;
        const lab = `${dlabel(w.time[i])} ${pad(h)}시 ${on ? '가동, 생산량 ' + num0(ED.prod[i]) : '비가동'}, 예측 ${fmt(pred[i], 0)}kW${risk ? ', 피크 위험' : ''}`;
        body += `<td><button type="button" class="${cls}" data-i="${i}" aria-label="${lab}" title="예측 ${fmt(pred[i], 0)}kW">${on ? num0(ED.prod[i]) : ''}</button></td>`;
      }
      body += '</tr>';
    }
    body += '</tbody>';
    let ft = '<tfoot><tr><th class="hr">합계</th>';
    for (let d = 0; d < n / 24; d++) {
      const r = v.days[d];
      ft += `<td><input type="number" min="0" step="100" inputmode="numeric" data-d="${d}" value="${Math.round(r.prod_before)}" class="${r.out_of_range ? 'oor' : ''}" aria-label="${dlabel(w.time[d * 24])} 하루 생산량 합계" title="${r.out_of_range ? '과거 하루 생산량 99% 수준보다 많음' : '하루 합계를 바꾸면 가동 시간에 비율대로 나눠요'}"></td>`;
    }
    ft += '</tr></tfoot>';
    const grid = $('#pe-grid');
    grid.innerHTML = hd + body + ft;
  }

  function edChanged(msg) {
    ED.res = null; ED.ver++;
    saveDraft();
    edCompute(); renderGrid(); renderPlanBody();
    if (msg) edStatus(msg);
  }
  const edMode = () => (document.querySelector('input[name="pe-mode"]:checked') || {}).value || 'toggle';

  function editProd(btn) {
    const i = +btn.dataset.i;
    const input = document.createElement('input');
    input.type = 'number'; input.min = '0'; input.step = '10'; input.value = ED.on[i] ? Math.round(ED.prod[i]) : 0;
    input.setAttribute('aria-label', '생산량');
    btn.textContent = ''; btn.appendChild(input); input.focus(); input.select();
    let done = false;
    const commit = (ok) => {
      if (done) return; done = true;
      const val = Math.max(0, +input.value || 0);
      if (ok) {
        ED.prod[i] = val;
        if (val > 0) ED.on[i] = 1;
        edChanged(`${tlabel(edWeek().time[i])} 생산량을 ${num0(val)}(으)로 바꿨어요.`);
      } else renderGrid();
    };
    input.onkeydown = (e) => { if (e.key === 'Enter') commit(true); if (e.key === 'Escape') commit(false); };
    input.onblur = () => commit(true);
    input.onclick = (e) => e.stopPropagation();
  }
  function toggleCell(i, val) {
    const t = tpl();
    ED.on[i] = val; ED.ver++;
    if (val === 1 && !(ED.prod[i] > 0)) ED.prod[i] = t.on[i] ? t.production[i] : 0;
  }

  function initEditor() {
    const grid = $('#pe-grid');
    grid.addEventListener('pointerdown', (e) => {
      const b = e.target.closest('.pe-cell');
      if (!b || edMode() !== 'toggle' || e.button !== 0) return;
      const i = +b.dataset.i;
      ED.paint = { val: ED.on[i] ? 0 : 1, cells: new Set([i]) };
      toggleCell(i, ED.paint.val);
      b.classList.toggle('off', ED.paint.val === 0);
      b.textContent = ED.paint.val ? num0(ED.prod[i]) : '';
      e.preventDefault();
    });
    grid.addEventListener('pointerover', (e) => {
      if (!ED.paint || e.pointerType !== 'mouse') return;
      const b = e.target.closest('.pe-cell');
      if (!b) return;
      const i = +b.dataset.i;
      if (ED.paint.cells.has(i)) return;
      ED.paint.cells.add(i);
      toggleCell(i, ED.paint.val);
      b.classList.toggle('off', ED.paint.val === 0);
      b.textContent = ED.paint.val ? num0(ED.prod[i]) : '';
    });
    window.addEventListener('pointerup', () => {
      if (!ED.paint) return;
      const k = ED.paint.cells.size, val = ED.paint.val;
      ED.paint = null;
      edChanged(`${k}시간을 ${val ? '가동' : '비가동'}으로 바꿨어요.`);
    });
    grid.addEventListener('click', (e) => {
      const b = e.target.closest('.pe-cell');
      if (b && edMode() === 'prod' && !b.querySelector('input')) editProd(b);
    });
    grid.addEventListener('keydown', (e) => {      // 키보드: Enter/Space = 지금 모드의 동작
      const b = e.target.closest('.pe-cell');
      if (!b || e.target.tagName === 'INPUT' || (e.key !== 'Enter' && e.key !== ' ')) return;
      e.preventDefault();
      const i = +b.dataset.i;
      if (edMode() === 'prod') editProd(b);
      else { toggleCell(i, ED.on[i] ? 0 : 1); edChanged(`${tlabel(edWeek().time[i])}을 ${ED.on[i] ? '가동' : '비가동'}으로 바꿨어요.`); $(`.pe-cell[data-i="${i}"]`)?.focus(); }
    });
    grid.addEventListener('change', (e) => {
      const inp = e.target.closest('tfoot input');
      if (!inp) return;
      const d = +inp.dataset.d, target = Math.max(0, +inp.value || 0);
      const ix = Array.from({ length: 24 }, (_, h) => d * 24 + h).filter((i) => ED.on[i]);
      if (!ix.length) { edStatus('그날은 가동 시간이 없어서 합계를 나눌 수 없어요. 먼저 가동할 시간을 켜 주세요.'); renderGrid(); return; }
      const cur = ix.reduce((s, i) => s + ED.prod[i], 0);
      ix.forEach((i) => { ED.prod[i] = cur > 0 ? Math.round(ED.prod[i] * target / cur) : Math.round(target / ix.length); });
      edChanged(`${dlabel(edWeek().time[d * 24])} 하루 생산량을 ${num0(target)}(으)로 맞췄어요 (가동 시간에 비율대로).`);
    });

    $('#pe-suggest').onclick = async () => {
      if (ED.busy) return;
      ED.busy = true;
      const btns = ['#pe-suggest', '#pe-apply', '#pe-reset', '#pe-file'].map((s) => $(s));
      btns.forEach((b) => { b.disabled = true; });
      edStatus('수정안을 계산하고 있어요… (후보마다 모델로 다시 예측해서 몇 초 걸려요)');
      // 계산을 시작한 시점의 계획·주·모드를 기억해 두고, 끝났을 때 그사이 바뀌었으면 결과를 버린다
      const ver = ED.ver, key = ED.key, ext = S.pext;
      const mode = PA.modes[ext ? 'ext' : 'base'], t0 = performance.now();
      try {
        const prodNow = ED.prod.map((v, i) => (ED.on[i] ? v : 0));
        const res = await ED.engine.search(key, ED.on.slice(), prodNow, {
          allowNew: mode.allow_new, maxMoveShare: mode.max_move_share,
          yield: () => new Promise((r) => setTimeout(r, 0)),
          onProgress: (p) => edStatus(`수정안을 계산하고 있어요… 후보 ${p.evals.toLocaleString('ko-KR')}개 평가, 이동 ${p.moves}건 채택`),
        });
        if (ED.ver !== ver || ED.key !== key) {
          edStatus('계산하는 동안 계획이 바뀌어서 이 수정안은 쓰지 않았어요. 다시 "수정안 만들기"를 눌러 주세요.');
          return;
        }
        ED.res = res;
        ED.resExt = ext;                                  // 계산에 쓴 모드 (지금 모드와 다르면 edCompute가 수정안을 쓰지 않음)
        edCompute(); renderGrid(); renderPlanBody();
        const s = ED.view.summary;
        edStatus(ED.res.moves.length
          ? `수정안이 나왔어요 (${((performance.now() - t0) / 1000).toFixed(1)}초): 주간 예측 최대 ${fmt(s.pred_week_max_before)} → <b>${fmt(s.pred_week_max_after)}kW</b>, 이동 ${s.n_moves}건. 아래 "무엇을 바꾸나요?"에서 확인하고, 마음에 들면 "수정안을 계획에 반영"을 누르세요.`
          : '규칙 안에서 최대전력을 의미 있게 낮추는 이동을 찾지 못했어요. 계획을 바꿔 다시 해 보세요.');
      } catch (err) {
        edStatus('계산 중 오류가 났어요: ' + err.message);
      } finally {
        ED.busy = false;
        btns.forEach((b) => { b.disabled = false; });
        $('#pe-apply').disabled = !(ED.res && ED.resExt === S.pext && ED.res.moves.length);
      }
    };
    $('#pe-apply').onclick = () => {
      if (!ED.res) return;
      ED.on = ED.res.on.slice(); ED.prod = ED.res.production.map((v) => Math.round(v));
      edChanged('수정안을 계획에 반영했어요. 여기서 더 고치거나 다시 수정안을 만들 수 있어요.');
    };
    $('#pe-reset').onclick = () => {
      const t = tpl();
      ED.on = t.on.slice(); ED.prod = t.production.slice();
      edChanged('처음 계획으로 되돌렸어요.');
    };
    $('#pe-share').onclick = () => {
      const payload = { k: ED.key, x: S.pext ? 1 : 0, c: diffNow() };
      const b64 = btoa(unescape(encodeURIComponent(JSON.stringify(payload)))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
      const url = location.href.split('#')[0] + '#plan&e=' + b64;
      const box = $('#pe-link');
      box.hidden = false; box.value = url;
      const ok = () => edStatus(`공유 링크를 복사했어요 (고친 칸 ${payload.c.length}개). 이 링크를 열면 같은 계획이 열려요.`);
      const fallback = () => { box.focus(); box.select(); edStatus('아래 링크를 선택했어요. Ctrl+C로 복사해 공유하세요.'); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(url).then(ok, fallback); else fallback();
    };
    $('#pe-csv').onclick = () => {
      const w = edWeek();
      const rows = ['date,hour,dow,on,production'].concat(w.time.map((t, i) => {
        const d = t.slice(0, 10);
        return `${d},${+t.slice(11, 13)},${DOW[dowOf(d)]},${ED.on[i]},${ED.on[i] ? Math.round(ED.prod[i]) : 0}`;
      }));
      const blob = new Blob(['﻿' + rows.join('\r\n') + '\r\n'], { type: 'text/csv;charset=utf-8' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob); a.download = `plan_${w.start}_edited.csv`;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
      edStatus(`${a.download}로 내려받았어요. 터미널에서 python 14_plan_advisor.py --plan 으로도 같은 결과가 나와요.`);
    };
    $('#pe-file').onchange = (e) => {
      const f = e.target.files[0];
      if (!f) return;
      f.text().then((txt) => {
        const lines = txt.replace(/^﻿/, '').split(/\r?\n/).filter((l) => l.trim());
        const head = lines[0].split(',').map((x) => x.trim());
        const ci = (n) => head.indexOf(n);
        if (ci('date') < 0 || ci('hour') < 0 || ci('on') < 0) throw new Error('date, hour, on 열이 있어야 해요');
        const w = edWeek(), pos = {};
        w.time.forEach((t, i) => { pos[`${t.slice(0, 10)} ${+t.slice(11, 13)}`] = i; });
        const on = ED.on.slice(), prod = ED.prod.slice();
        let n = 0;
        lines.slice(1).forEach((l) => {
          const c = l.split(',');
          const i = pos[`${c[ci('date')].trim()} ${+c[ci('hour')]}`];
          if (i === undefined) return;
          on[i] = +c[ci('on')] ? 1 : 0;
          const p = ci('production') >= 0 ? c[ci('production')] : '';
          prod[i] = on[i] ? (p === undefined || p.trim() === '' ? (tpl().on[i] ? tpl().production[i] : 0) : Math.max(0, +p || 0)) : 0;
          n++;
        });
        if (n !== w.time.length) throw new Error(`이 주(${dlabel(w.start)}~${dlabel(w.end)})의 168시간이 모두 있어야 해요. 맞는 행 ${n}개`);
        ED.on = on; ED.prod = prod;
        edChanged(`${f.name}을 불러왔어요.`);
      }).catch((err) => edStatus('CSV를 불러오지 못했어요: ' + err.message)).finally(() => { e.target.value = ''; });
    };
  }

  // 편집기 주를 고르면 데이터를 불러와 표와 아래 결과를 그린다
  function renderEditor() {
    $('#p-editor').hidden = false;
    if (!ED.engine) {
      $('#p-lead').innerHTML = '편집기를 불러오는 중이에요… (모델 약 2MB)';
      ['#p-kpis', '#p-chart', '#p-moves', '#p-days'].forEach((s) => { $(s).innerHTML = ''; });
      $('#pe-grid').innerHTML = '';
      loadEditor().then(() => { if (isEdit()) renderPlan(); })
        .catch((err) => { $('#p-lead').textContent = '편집기를 불러오지 못했어요: ' + err.message + ' (14_plan_advisor.py --web-editor 로 web/data/plan_editor.js 를 만들어야 해요)'; });
      return false;
    }
    if (ED.key !== S.pweek) openEditor(S.pweek);
    edCompute(); renderGrid();
    return true;
  }

  function renderPlanKpis() {
    const s = pw().summary;
    if (pw().pending) {   // 직접 편집에서 아직 수정안을 만들기 전: 지금 계획의 예측만
      $('#p-kpis').innerHTML = `
        <div class="kpi"><div class="label">지금 계획의 주간 예측 최대</div>
          <div class="value">${fmt(s.pred_week_max_before)}<span class="from">kW</span></div>
          <div class="delta">수정안 전</div><div class="foot">1주 전 예측 · 요금 기준이 되는 값</div></div>
        <div class="kpi"><div class="label">피크 위험 시간</div>
          <div class="value">${s.risk_hours_before}<span class="from">시간</span></div>
          <div class="delta">예측 + 마진 ${fmt(s.margin, 0)}kW ≥ ${fmt(s.peak_threshold, 0)}kW</div><div class="foot">표에서 주황 표시된 칸</div></div>
        <div class="kpi"><div class="label">주간 계획 생산량</div>
          <div class="value">${num0(s.total_production)}</div>
          <div class="delta">${s.out_of_range_days.length ? `범위 밖 ${s.out_of_range_days.length}일` : '학습 범위 안'}</div><div class="foot">하루 ${num0(s.day_prod_p99)} 넘으면 예측을 믿기 어려움</div></div>
        ${s.backtest ? `<div class="kpi"><div class="label">실제 피크 시간 (${fmt(s.peak_threshold, 0)}kW 이상)</div>
          <div class="value">${s.actual_peak_hours}<span class="from">시간</span></div><div class="delta">비교용 실제값</div><div class="foot">실제 주간 최대 ${fmt(s.actual_week_max, 0)}kW</div></div>` : ''}`;
      return;
    }
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
      k += `<div class="kpi"><div class="label">기본요금 절감 ${tariffTag()}</div>
        <div class="value"><span class="cu" data-from="0" data-to="${Math.round(month)}" data-dec="0" data-suf="원">${won(month)}</span><span class="from">/월</span></div>
        <div class="delta">연 ${won(month * 12)}</div><div class="foot">${tariffFoot('주간 예측 최대 절감')}</div></div>`;
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
    s += `<path d="${path(H0.pred_before)}" fill="none" style="stroke:var(--orange)" stroke-width="2.5" stroke-linejoin="round"/>`;
    s += `<path d="${path(H0.pred_after)}" fill="none" style="stroke:var(--blue)" stroke-width="2.75" stroke-linejoin="round"/>`;
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
    if (!w.moves.length) {
      $('#p-moves').innerHTML = w.pending
        ? '<p class="empty">아직 수정안을 만들지 않았어요. 위 표에서 계획을 고친 뒤 <b>수정안 만들기</b>를 누르세요.</p>'
        : '<p class="empty">규칙 안에서 최대전력을 의미 있게 낮추는 이동을 찾지 못했습니다.</p>';
      return;
    }
    const ordered = w.moves.slice().sort((a, b) => (a.kind === b.kind ? 0 : a.kind === 'day' ? -1 : 1));
    $('#p-moves').innerHTML = ordered.map((m, i) => {
      const f = m.from.slice(0, 10), t = m.to.slice(0, 10), fd = days[f], td = days[t];
      if (m.kind === 'day') {
        const pct = fd.prod_before ? Math.round(100 * m.qty / fd.prod_before) : 0;
        const ext = m.new_hours.length ? `<p class="why"><span class="tag exp">실험적</span> ${dlabel(t)} ${m.new_hours.map((h) => h.slice(11, 13) + '시').join(', ')}를 새로 가동해서 받습니다. 과거에 이 요일·시각에 가동한 선례가 있는 시간만 골랐지만, 선례가 적어 예측을 그대로 믿기 어렵습니다.</p>` : '';
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
    const rows = w.days.map((r) => `<tr><td>${dlabel(r.date)}${r.out_of_range ? ' <span class="tag" title="계획 생산량이 학습 범위 밖">범위 밖</span>' : ''}</td><td class="num">${r.on_before} → ${r.on_after}</td>
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
        <li>결과: plans/out/…/advice.md, revised_plan.csv (수정된 계획)</li>
        <li>이 탭에 올리기: 같은 명령 끝에 <code>--publish 이름</code>을 붙이면 웹 데이터까지 다시 만들어 '주' 목록에 '내 계획 · 이름'으로 나옵니다. 내릴 때는 <code>--unpublish 이름</code></li></ul>`);
  }

  function renderPlan() {
    if (!PA) {
      $('#view-plan').innerHTML = '<p class="card">주간 계획 데이터가 없습니다. 13_week_ahead.py → 14_plan_advisor.py --web-examples → 12_build_report.py 순서로 실행하세요.</p>';
      return;
    }
    if (isEdit()) { if (!renderEditor()) return; } else $('#p-editor').hidden = true;
    renderPlanBody();
  }
  function renderPlanBody() {
    const w = pw(), s = w.summary;
    const bt = s.backtest ? ` 시작일 이전 전력만 써서 예측했고, 실제 전력은 비교용입니다.` : '';
    if (w.source === 'edit') {
      $('#p-lead').innerHTML = `위 표에서 <b>${dlabel(s.start)}~${dlabel(s.end)}</b> 계획을 직접 고쳐 보세요. 예측과 수정안은 브라우저에서 바로 계산하고, 터미널(14_plan_advisor.py)과 같은 결과가 나와요.${bt}${w.pending ? ' <b>아직 수정안을 만들지 않았어요</b> — 아래 수치는 지금 계획 그대로의 예측이에요.' : ''}`;
    } else if (w.source === 'user') {
      $('#p-lead').innerHTML = `<b>직접 넣은 계획</b>(${w.label.replace(/^내 계획 · /, '')}, ${dlabel(s.start)}~${dlabel(s.end)})의 수정안입니다.${bt}`;
    } else {
      $('#p-lead').innerHTML = s.backtest
        ? `${dlabel(s.start)}~${dlabel(s.end)}에 <b>실제로 실행된 계획</b>을 넣어 본 백테스트입니다.${bt}`
        : `데이터 다음 주(${dlabel(s.start)}~${dlabel(s.end)})의 <b>평소 일정 계획</b>(요일별 가동 + 최근 4주 평균 생산량)을 넣은 예시입니다. 내 계획 CSV를 올리는 방법은 아래 '내 계획을 넣어 보려면'에 있습니다.`;
    }
    const oor = s.out_of_range_days || [];
    if (oor.length) {
      $('#p-lead').innerHTML += `<span class="callout" style="display:block;margin-top:10px"><strong>주의:</strong> ${oor.map(dlabel).join(', ')}은 계획 생산량이 과거 하루 생산량의 99% 수준(${num0(s.day_prod_p99)})보다 많습니다. 학습에 거의 없던 물량이라 모델이 생산량 변화에 둔감해서, 이 날의 예측과 수정안은 믿기 어렵습니다.</span>`;
    }
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
    if (v === 'chat') updateCtx();
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
    $('#a-show-move').checked = S.showMove;
    $('#a-show-hs').checked = S.showHs;
    $('#a-show-move').onchange = (e) => { S.showMove = e.target.checked; store.set('showMove', S.showMove ? '1' : '0'); renderChart(); };
    $('#a-show-hs').onchange = (e) => { S.showHs = e.target.checked; store.set('showHs', S.showHs ? '1' : '0'); renderChart(); };
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
    // 주소 끝: #보기[&e=공유한 계획]
    const [h, ...params] = location.hash.slice(1).split('&');
    const share = params.find((x) => x.startsWith('e='));
    if (share) {
      try {
        const b64 = share.slice(2).replace(/-/g, '+').replace(/_/g, '/');
        S.shareEdit = JSON.parse(decodeURIComponent(escape(atob(b64 + '==='.slice((b64.length + 3) % 4)))));
      } catch (e) { S.shareEdit = null; }
    }
    initWeeks();
    initChat();
    initPlan();
    setView(S.shareEdit ? 'plan' : ['admin', 'worker', 'plan', 'chat'].includes(h) ? h : 'admin');
    window.addEventListener('hashchange', () => {
      const n = location.hash.slice(1).split('&')[0];
      if (['admin', 'worker', 'plan', 'chat'].includes(n) && n !== S.view) setView(n);
    });
  }
  init();
})();
