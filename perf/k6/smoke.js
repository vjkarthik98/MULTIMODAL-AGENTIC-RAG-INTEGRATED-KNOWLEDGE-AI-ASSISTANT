// Smoke test — sanity-check the script + auth + a real query end to end
// before running stress.js/soak.js at scale. Local target by default.
//
// Usage:
//   docker compose up -d api qdrant redis mongo
//   python -m app.bin.seed_test_tenants
//   export MAGIK_TEST_TENANTS="$(cat .magik_test_tenants.json)"
//   k6 run perf/k6/smoke.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { BASE_URL, authHeaders, login, tenantForVU } from './lib/auth.js';

export const options = {
  vus: 2,
  iterations: 4,
  thresholds: {
    http_req_failed: ['rate<0.05'],
    'http_req_duration{name:health}': ['p(95)<1000'],
  },
};

let token;

export default function () {
  if (!token) {
    token = login(tenantForVU());
  }

  const health = http.get(`${BASE_URL}/health`, { tags: { name: 'health' } });
  check(health, { 'health 200': (r) => r.status === 200 });

  const query = http.post(
    `${BASE_URL}/rag/query`,
    JSON.stringify({
      query: 'What is this knowledge base about?',
      session_id: `k6-smoke-${__VU}-${__ITER}`,
      no_cache: true,
    }),
    Object.assign({ tags: { name: 'rag_query' } }, authHeaders(token))
  );
  check(query, {
    'query 200': (r) => r.status === 200,
    'query has answer field': (r) => {
      try {
        return typeof r.json('answer') === 'string';
      } catch (e) {
        return false;
      }
    },
  });

  sleep(1);
}
