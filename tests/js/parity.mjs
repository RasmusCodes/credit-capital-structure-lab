// Node runner for the JavaScript/Python parity check (same checks as web/test/index.html).
// Usage: node tests/js/parity.mjs   (CI runs it; vectors come from scripts/export_web.py)
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import * as E from "../../web/lab-engine.js";

const here = dirname(fileURLToPath(import.meta.url));
const v = JSON.parse(readFileSync(join(here, "..", "..", "web", "test", "vectors.json"), "utf8"));
let n = 0, bad = 0;
const close = (a, b, rel, abs = 0) => Math.abs(a - b) <= Math.max(abs, rel * Math.max(Math.abs(a), Math.abs(b)));
function check(name, a, b, rel, abs = 0) { n++; if (!close(a, b, rel, abs)) { bad++; console.log(`FAIL ${name}: js ${a} py ${b}`); } }

v.normcdf.x.forEach((x, i) => check(`normCdf(${x})`, E.normCdf(x), v.normcdf.y[i], 1e-12, 1e-15));
for (const key of ["waterfall_coreweave", "waterfall_lbo_gift"]) {
  const w = v[key];
  w.V.forEach((V, i) => {
    const r = E.distribute(w.structure, V, { gift: w.gift || 0 });
    for (const [id, arr] of Object.entries(w.payouts)) check(`${key} ${V} ${id}`, r.payouts[id], arr[i], 1e-9, 1e-9);
  });
}
const lbo = v.waterfall_lbo_gift.structure;
const bp = E.breakpoints(lbo, { inDefault: true });
v.breakpoints_lbo.in_default.forEach((x, i) => check(`breakpoint ${i}`, bp[i], x, 1e-6));
for (const m of ["merton", "black_cox"]) {
  const t = v[`pricing_lbo_${m}`];
  const pr = E.priceStructure(lbo, { V0: t.V0, sigma: t.sigma, T: t.T, r: t.r, q: 0, model: m, H0: t.H0, defaultPoint: t.default_point });
  for (const [id, val] of Object.entries(t.values)) check(`price ${m} ${id}`, pr.values[id], val, 1e-6, 1e-6);
  check(`pd ${m}`, pr.pd, t.pd, 1e-6, 1e-9);
}
check("firstPassageProb", E.firstPassageProb(...v.first_passage.args), v.first_passage.p, 1e-10);
check("discountedHit", E.discountedHit(...v.discounted_hit.args), v.discounted_hit.v, 1e-10);
const bm = E.barrierPartialMoments(...v.barrier_moments.args);
check("barrier p", bm.p, v.barrier_moments.p, 1e-10); check("barrier ev", bm.ev, v.barrier_moments.ev, 1e-10);
const pm = E.partialMoments(...v.partial_moments.args);
check("partial p", pm.p, v.partial_moments.p, 1e-10); check("partial ev", pm.ev, v.partial_moments.ev, 1e-10);
const [V0, s, H0, g, r, q] = v.pd_term.args;
E.pdTermStructure(V0, s, H0, g, r, q, v.pd_term.times).forEach((x, i) => check(`pdTerm ${i}`, x, v.pd_term.pd[i], 1e-10));
console.log(bad ? `FAIL ${bad}/${n}` : `PASS ${n}/${n}`);
process.exit(bad ? 1 : 0);
