// Soak test — sustained low, constant load over a long duration to catch
// memory/connection leaks and latency drift that only show up over time
// (GPU VRAM fragmentation, unclosed Qdrant/Redis/Mongo connections, growing
// per-session state) — the kind of bug a short stress.js burst never
// surfaces. Local target by default.
//
// Default duration is 30 minutes at a constant 3 VUs — long enough to see a
// drift trend, short enough to run in a normal dev session. For a real
// overnight soak, override with -e SOAK_DURATION=8h.
//
// Usage:
//   docker compose up -d api qdrant redis mongo
//   python -m app.bin.seed_test_tenants --count 5
//   export MAGIK_TEST_TENANTS="$(cat .magik_test_tenants.json)"
//   k6 run perf/k6/soak.js
//   k6 run -e SOAK_DURATION=8h perf/k6/soak.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend } from 'k6/metrics';
import { BASE_URL, authHeaders, login, tenantForVU } from './lib/auth.js';

export const queryLatency = new Trend('rag_query_duration', true);

export const options = {
  vus: 3,
  duration: __ENV.SOAK_DURATION || '30m',
  thresholds: {
    // The interesting soak signal isn't a single threshold — it's whether
    // p95 measured in the LAST 10 minutes of the run is materially worse
    // than the first 10 (compare rag_report-style JSON summary export
    // across two soak runs). This threshold just guards against outright
    // breakage during the run.
    http_req_failed: ['rate<0.02'],
  },
};

let token;

export default function () {
  if (!token) {
    token = login(tenantForVU());
  }

  const res = http.post(
    `${BASE_URL}/rag/query`,
    JSON.stringify({
      query: 'Summarize the key financial highlights.',
      session_id: `k6-soak-${__VU}`, // stable session across iterations — exercises memory-layer growth, not just per-request cost
      no_cache: true,
    }),
    Object.assign({ tags: { name: 'rag_query' } }, authHeaders(token))
  );
  check(res, { 'not a 5xx': (r) => r.status < 500 });
  if (res.status === 200) {
    queryLatency.add(res.timings.duration);
  }

  sleep(10); // low, constant rate — this is about duration, not throughput
}
