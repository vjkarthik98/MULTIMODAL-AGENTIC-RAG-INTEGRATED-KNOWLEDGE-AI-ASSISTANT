// Live-mode load profile — manual, on-demand only, against the REAL deployed
// MAGIK instance. This is deliberately NOT wired into any scheduled or
// on-push CI workflow (see .github/workflows/quality-live.yml,
// workflow_dispatch-only) — running it wakes the wake-on-demand AWS box
// (deploy/aws/lambda/wake_gateway/) and holds it awake for the run's
// duration. Accepted, deliberate cost for a portfolio demo with low real
// traffic (per the approved plan), but always a conscious choice.
//
// Scale is deliberately modest: this profile represents realistic portfolio
// traffic (a handful of recruiters looking at once), not an attempt to find
// the breaking point of the one production box — that's stress.js's job,
// and it stays local-only. Also note RATE_LIMIT_RPM=60/min is enforced per
// CLIENT IP (see lib/auth.js) — a handful of VUs from one host will
// realistically brush against that ceiling, which is itself useful,
// authentic signal about the live system.
//
// Usage:
//   python -m app.bin.seed_test_tenants --count 5   # once, reuse afterwards
//   export MAGIK_TEST_TENANTS="$(cat .magik_test_tenants.json)"
//   export MAGIK_API_BASE_URL=https://magik.vk-ai.online
//   k6 run perf/k6/live_profile.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';
import { BASE_URL, authHeaders, login, tenantForVU } from './lib/auth.js';

export const rateLimited = new Rate('rate_limited_429');
export const serverErrors = new Rate('server_errors_5xx');
export const queryLatency = new Trend('rag_query_duration', true);

export const options = {
  stages: [
    { duration: '20s', target: 3 },
    { duration: '1m', target: 5 }, // "a few recruiters looking at once"
    { duration: '20s', target: 0 },
  ],
  thresholds: {
    server_errors_5xx: ['rate<0.01'],
  },
};

let token;

export default function () {
  if (!token) {
    console.log(`[live_profile] target: ${BASE_URL} — this run wakes the box if it's asleep`);
    token = login(tenantForVU());
  }

  const health = http.get(`${BASE_URL}/health`, { tags: { name: 'health' } });
  serverErrors.add(health.status >= 500);

  const res = http.post(
    `${BASE_URL}/rag/query`,
    JSON.stringify({
      query: 'What financial documents are in this knowledge base?',
      session_id: `k6-live-${__VU}-${__ITER}`,
      no_cache: true,
    }),
    Object.assign({ tags: { name: 'rag_query' } }, authHeaders(token))
  );
  rateLimited.add(res.status === 429);
  serverErrors.add(res.status >= 500);
  if (res.status === 200) {
    queryLatency.add(res.timings.duration);
  }
  check(res, { 'not a 5xx': (r) => r.status < 500 });

  sleep(Math.random() * 3 + 2); // slower think time than stress.js — this mimics real browsing, not load-finding
}
