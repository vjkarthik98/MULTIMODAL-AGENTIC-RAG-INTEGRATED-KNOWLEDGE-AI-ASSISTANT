# Browser Performance Reports (Lighthouse)

[Lighthouse](https://github.com/GoogleChrome/lighthouse) / [Lighthouse CI](https://github.com/GoogleChrome/lighthouse-ci)
(Apache-2.0). Two modes, same tool family:

## Local (automatic, every PR touching `ui/`)

`@lhci/cli` builds `ui/dist` and serves it locally — zero cost, deterministic,
no live infrastructure touched. Config: `lighthouserc.json` (repo root).

```bash
make lighthouse
# or directly:
npm --prefix ui run build
npx --yes @lhci/cli@0.14 autorun --config=lighthouserc.json
```

Raw per-run output lands in `quality-reports/browser-performance/.lhci/`
(gitignored — regenerable noise, not worth committing). This directory's
own dated `*.md` summaries (see `scripts/generate_quality_badges.py`) are
what's committed and linked from the README.

## Live (manual, on-demand — the portfolio's real Core Web Vitals number)

The local pass above measures the bundle in isolation — no real network
latency, no real Caddy/TLS overhead, no real cold vs. warm instance
difference. For the number worth publishing, run plain `lighthouse`
(not `lhci`) against the real deployed URL:

```bash
npx --yes lighthouse https://magik.vk-ai.online \
  --output=html --output=json \
  --output-path=quality-reports/browser-performance/$(date -u +%Y%m%d-%H%M%S)-live \
  --only-categories=performance,accessibility,best-practices \
  --chrome-flags="--headless"
```

This wakes the wake-on-demand AWS box if it's asleep — run it deliberately,
not on a schedule. The first-run number will include the wake/cold-start
interstitial; run it a second time immediately after for the "warm" number,
and report both — the cold number is honest signal about the wake-on-demand
architecture, not a flaw to hide.
