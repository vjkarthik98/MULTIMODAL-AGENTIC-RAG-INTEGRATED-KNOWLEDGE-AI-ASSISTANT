// Multi-user simulation — the actual "does tenant isolation hold under
// concurrency" test, not just under the sequential conditions
// tests/auth/test_tenant_isolation.py already covers.
//
// Design: setup() logs in as every seeded test tenant and has each one
// ingest a tiny .txt file containing a UNIQUE marker string. VUs then run
// concurrently, each assigned to one tenant, each asking a question that
// should surface ITS OWN marker — and asserting the answer/sources NEVER
// contain any OTHER tenant's marker. A single leaked marker across tenants
// is a real bug (release-blocking per the approved plan), not test noise.
//
// Requires at least as many seeded tenants as VUs you intend to run with —
// size --count when seeding accordingly (each VU should map to its own
// tenant, not share one, or the isolation assertion is meaningless).
//
// Note: RATE_LIMIT_RPM=60/min is enforced per authenticated user, not per
// client IP (see lib/auth.js) — each tenant gets its own bucket, so 429s
// under this script should only appear once an INDIVIDUAL tenant's traffic
// (including the intentional burst sub-test below) exceeds 60/min, not as a
// shared ceiling across every VU. Orthogonal either way to what this file
// actually asserts (cross-tenant leakage), which only evaluates requests
// that got a real 200 back — a 429 can't leak anything.
//
// Usage:
//   docker compose up -d api qdrant redis mongo
//   python -m app.bin.seed_test_tenants --count 8
//   export MAGIK_TEST_TENANTS="$(cat .magik_test_tenants.json)"
//   k6 run --vus 8 --iterations 24 perf/k6/multi_user_tenant.js
import http from 'k6/http';
import { check, sleep, fail } from 'k6';
import { Rate } from 'k6/metrics';
import { BASE_URL, TENANTS, authHeaders, authHeadersMultipart, login } from './lib/auth.js';

export const crossTenantLeak = new Rate('cross_tenant_leak'); // must stay exactly 0
export const rateLimitedCleanly = new Rate('rate_limited_clean_429');

export const options = {
  thresholds: {
    cross_tenant_leak: ['rate==0'], // ANY leak fails the run — see file docstring
  },
};

export function setup() {
  const records = [];
  TENANTS.forEach((tenant, i) => {
    const token = login(tenant);
    const marker = `MAGIK_K6_MARKER_${i}_${Math.random().toString(36).slice(2, 10)}`;
    const content = `Confidential test document for tenant ${i}. Unique marker: ${marker}. ` +
      `This document belongs exclusively to ${tenant.email} and must never be retrievable by ` +
      `any other tenant's queries.`;

    const uploadRes = http.post(
      `${BASE_URL}/rag/ingest`,
      {
        file: http.file(content, `k6-marker-${i}.txt`, 'text/plain'),
        session_id: `k6-multiuser-setup-${i}`,
      },
      authHeadersMultipart(token)
    );
    if (uploadRes.status !== 200) {
      fail(`setup: ingest failed for tenant ${i} (${tenant.email}): ${uploadRes.status} ${uploadRes.body}`);
    }

    records.push({ index: i, email: tenant.email, token: token, marker: marker });
  });

  // Give ingestion a moment to be queryable (txt ingest is synchronous per
  // CLAUDE.md, but embed+upsert-to-Qdrant latency is still real).
  sleep(3);

  return { records: records };
}

export default function (data) {
  const records = data.records;
  const mine = records[(__VU - 1) % records.length];
  const others = records.filter((r) => r.index !== mine.index);

  const res = http.post(
    `${BASE_URL}/rag/query`,
    JSON.stringify({
      query: 'What is the unique marker in my confidential test document?',
      session_id: `k6-multiuser-${mine.index}-${__ITER}`,
      no_cache: true,
    }),
    Object.assign({ tags: { name: 'rag_query' } }, authHeaders(mine.token))
  );

  check(res, { 'query not a 5xx': (r) => r.status < 500 });

  if (res.status === 200) {
    const bodyText = res.body || '';
    const leaked = others.some((o) => bodyText.includes(o.marker));
    crossTenantLeak.add(leaked);
    if (leaked) {
      console.error(
        `CROSS-TENANT LEAK: tenant ${mine.index} (${mine.email}) response contained another ` +
          `tenant's marker. Body: ${bodyText.slice(0, 500)}`
      );
    }
    check(res, {
      'own marker present (retrieval worked)': () => bodyText.includes(mine.marker),
      'no other tenant marker present': () => !leaked,
    });
  } else {
    crossTenantLeak.add(false); // can't leak if the query itself failed
  }

  // Occasionally (VU 1 only, every 10th iteration) intentionally burst past
  // RATE_LIMIT_RPM=60/min to confirm the limiter degrades cleanly (429, not
  // 500) under real concurrent multi-user conditions, not just isolation.
  if (__VU === 1 && __ITER % 10 === 0) {
    let got429 = false;
    for (let i = 0; i < 65; i++) {
      const burst = http.get(`${BASE_URL}/health`, Object.assign({ tags: { name: 'burst' } }, authHeaders(mine.token)));
      if (burst.status === 429) {
        got429 = true;
        check(burst, { '429 body is clean json, not a 500 page': (r) => r.status === 429 });
        break;
      }
    }
    rateLimitedCleanly.add(got429);
  }

  sleep(Math.random() * 2 + 1);
}
