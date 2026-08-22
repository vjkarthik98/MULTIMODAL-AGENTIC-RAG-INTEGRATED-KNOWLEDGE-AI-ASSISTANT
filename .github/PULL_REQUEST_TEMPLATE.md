## What changed and why

<!-- The "why" matters more than the "what" — the diff already shows what changed. -->

## Type of change

- [ ] `feat` — new functionality
- [ ] `fix` — bug fix
- [ ] `refactor` — no behavior change
- [ ] `docs` — documentation only
- [ ] `ci` / `deps` / `chore`

## Checklist

- [ ] `make lint` passes
- [ ] `make test-unit` passes (plus any subsystem-specific suite touched — `test-auth`, `test-guardrails`, etc.)
- [ ] If this touches ingestion/chunking/embedding/retrieval: ran `make eval-retrieval` and checked for regressions
- [ ] If this touches a guardrail, auth, or tenant-isolation path: added or updated a test that would fail without the fix
- [ ] `.env.example` updated if a new setting was added to `Settings`
- [ ] `CHANGELOG.md` updated (Added / Fixed / Removed) if this is user- or operator-visible

## Related issue

Closes #
