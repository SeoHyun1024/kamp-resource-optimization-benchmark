// 브라우저 엔진(web/plan_engine.js)이 파이썬(14_plan_advisor.py)과 같은 예측·수정안을 내는지 확인한다.
// 실행: node check_plan_engine.mjs   (먼저 python 14_plan_advisor.py --web-editor)
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const ctx = { window: {} };
vm.runInNewContext(fs.readFileSync(path.join(here, 'web', 'data', 'plan_editor.js'), 'utf8'), ctx);
const data = ctx.window.PLAN_EDITOR_DATA;
const checks = JSON.parse(fs.readFileSync(path.join(here, 'plans', 'out', 'web', 'editor_check.json'), 'utf8'));
const engine = require('./web/plan_engine.js').create(data);

let ok = true;
const maxDiff = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));
for (const [name, ref] of Object.entries(checks)) {
  const ext = name.endsWith('_ext'), key = name.replace(/_ext$/, '');
  const w = data.weeks[key], mode = data.modes[ext ? 'ext' : 'base'];
  const t0 = Date.now();
  const p0 = engine.predict(key, w.template.on, w.template.production);
  const res = await engine.search(key, w.template.on, w.template.production, { allowNew: mode.allow_new, maxMoveShare: mode.max_move_share });
  const ms = Date.now() - t0;
  const d0 = maxDiff(Array.from(p0), ref.pred0), d1 = maxDiff(res.pred1, ref.pred1), dp = maxDiff(res.production, ref.production);
  const sameMoves = res.moves.length === ref.moves.length && res.moves.every((m, i) =>
    m.kind === ref.moves[i].kind && m.from === ref.moves[i].from && m.to === ref.moves[i].to &&
    Math.abs(m.qty - ref.moves[i].qty) < 1e-6 && JSON.stringify(m.new_hours) === JSON.stringify(ref.moves[i].new_hours));
  const sameOn = res.on.every((v, i) => v === ref.on[i]);
  const pass = d0 < 1e-9 && d1 < 1e-9 && dp < 1e-6 && sameMoves && sameOn;
  ok = ok && pass;
  console.log(`${pass ? 'OK ' : 'NG '} ${name}: 예측 차이 ${d0.toExponential(1)} / 수정 후 ${d1.toExponential(1)}, 이동 ${res.moves.length}건(${sameMoves ? '같음' : '다름'}), ${ms}ms`);
  if (!sameMoves) {
    console.log('  JS  ', res.moves.map((m) => `${m.kind} ${m.from}->${m.to} ${m.qty.toFixed(1)}`).join(' | '));
    console.log('  PY  ', ref.moves.map((m) => `${m.kind} ${m.from}->${m.to} ${m.qty.toFixed(1)}`).join(' | '));
  }
}
console.log(ok ? '모두 일치' : '불일치 있음');
process.exit(ok ? 0 : 1);
