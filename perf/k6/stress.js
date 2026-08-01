// Stress test — ramps VUs up to find where MAGIK's response time/error rate
// degrades. Local target by default (docker-compose); see live_profile.js for
// the separate, deliberately-smaller manual live-mode run.
//
// Stages are modest by design: this is a portfolio demo on a single GPU box,
// not a system being sized for real production traffic, so the goal is "how
// does it degrade" and "does it degrade cleanly (429s, not 500s/crashes)",
// not "how many thousands of RPS can it take". Real usage is a handful of
// recruiters at a time.
//
// VUs round-robin across the seeded test-tenant pool (perf/k6/lib/auth.js) —
// RATE_LIMIT_RPM is enforced per authenticated user (see lib/auth.js's note),
// so each VU genuinely gets its own 60/min bucket regardless of how many VUs
// share a source IP. 429s should therefore only show up per-VU once THAT
// tenant's own traffic exceeds 60/min, not as a whole-run ceiling shared by
// everyone — worth comparing against an earlier run's numbers if you have
// them, since this changed after the rate limiter was fixed to be per-user.
// 429s are tracked as their own metric, not folded into generic failures,
// since a 429 is the rate limiter working correctly, not a bug.
//
// Usage:
//   docker compose up -d api qdrant redis mongo
//   python -m app.bin.seed_test_tenants --count 10
//   export MAGIK_TEST_TENANTS="$(cat .magik_test_tenants.json)"
//   k6 run perf/k6/stress.js
//   # with results piped into the existing Grafana/Prometheus stack:
//   k6 run --out experimental-prometheus-rw perf/k6/stress.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';
import { BASE_URL, authHeaders, login, tenantForVU } from './lib/auth.js';

export const rateLimited = new Rate('rate_limited_429');
export const serverErrors = new Rate('server_errors_5xx');
export const queryLatency = new Trend('rag_query_duration', true);

export const options = {
  stages: [
    { duration: '30s', target: 5 }, // ramp-up
    { duration: '1m', target: 15 }, // realistic portfolio-visitor burst
    { duration: '30s', target: 25 }, // push past that to see degradation
    { duration: '30s', target: 5 }, // recovery
    { duration: '15s', target: 0 },
  ],
  thresholds: {
    server_errors_5xx: ['rate<0.01'], // near-zero 5xx is the real bar — 429s are expected, not failures
    'rag_query_duration': ['p(95)<60000'], // matches the README's documented p95 latency ceiling
  },
};

let token;

function weightedEndpoint() {
  // Cheap-endpoint-heavy mix — realistic browsing behavior generates far
  // more health/status/read calls than actual LLM queries, and it keeps the
  // one GPU box from being 100% saturated by generation calls alone.
  const r = Math.random();
  if (r < 0.5) return 'health';
  if (r < 0.75) return 'status';
  return 'query';
}

export default function () {
  if (!token) {
    token = login(tenantForVU());
  }

  const choice = weightedEndpoint();

  if (choice === 'health') {
    const res = http.get(`${BASE_URL}/health`, { tags: { name: 'health' } });
    check(res, { '200': (r) => r.status === 200 });
    serverErrors.add(res.status >= 500);
  } else if (choice === 'status') {
    const res = http.get(`${BASE_URL}/status`, Object.assign({ tags: { name: 'status' } }, authHeaders(token)));
    check(res, { 'ok': (r) => r.status === 200 || r.status === 401 });
    serverErrors.add(res.status >= 500);
  } else {
    const res = http.post(
      `${BASE_URL}/rag/query`,
      JSON.stringify({
        query: 'What was the reported revenue in the most recent filing?',
        session_id: `k6-stress-${__VU}-${__ITER}`,
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
  }

  sleep(Math.random() * 2 + 1); // 1-3s think time between actions
}
