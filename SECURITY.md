# Security Policy

MAGIK treats security as a first-class requirement, not an afterthought — see
the [Security & Guardrails](README.md#security--guardrails) section of the
README for what's built in (prompt-injection defense, tenant isolation,
authentication, secrets management) and `.github/workflows/security.yml` for
what runs automatically on every push (secret scanning, dependency CVE
scanning, SAST, container scanning).

This file covers the process for reporting a vulnerability *in the project
itself* — not the product security features documented elsewhere.

## Supported Versions

Only the latest tagged release of the current major version is supported.
There is no maintained backport branch: fixes land on `development`, ship in
the next tag, and are not backported to earlier ones.

| Version | Supported |
|---|---|
| 1.0.x (latest) | ✅ |
| < 1.0.0 | ❌ |

Pre-1.0.0 tags and the `1.0.0-rcN` candidates are development history and
receive no security fixes. Versioning follows
[Semantic Versioning](https://semver.org/).

## Reporting a Vulnerability

**Do not open a public GitHub issue for a security vulnerability.** Public
issues are appropriate for bugs that don't have exploit implications;
anything with a plausible attack path should be reported privately first.

Report privately via one of:

- **GitHub Security Advisories** (preferred): open a
  [private advisory](https://github.com/vjkarthik98/MULTIMODAL-AGENTIC-RAG-INTEGRATED-KNOWLEDGE-AI-ASSISTANT/security/advisories/new)
  on this repository.
- **Email**: karthikvj398@gmail.com — include "SECURITY" in the subject line.

Please include:
- A description of the vulnerability and its potential impact.
- Steps to reproduce (a minimal repro is ideal).
- Any relevant logs, requests, or payloads (redact real credentials/PII).

### What to expect

- **Acknowledgement**: within 5 business days.
- **Triage**: I'll confirm whether it's in scope and share an initial severity
  assessment.
- **Fix timeline**: depends on severity — critical issues (auth bypass,
  tenant-isolation break, injection bypass, secret exposure) are prioritized
  immediately; lower-severity issues are scheduled into the next release.
- **Disclosure**: coordinated — please allow a fix to ship before any public
  write-up. Credit is given in the CHANGELOG unless you'd prefer to stay
  anonymous.

## Scope

**In scope:**
- The application code in this repository (`app/`, `ui/`, ingestion/retrieval/
  generation pipelines, auth, guardrails).
- Infrastructure-as-code in `deploy/` and CI/CD workflows in `.github/workflows/`.
- Dependency vulnerabilities introduced by this project's pinned versions.

**Out of scope:**
- The live demo instance's availability (`magik.vk-ai.online` runs on a
  cost-optimized, scale-to-zero single GPU box by design — see
  [Deployment](README.md#deployment) — and is not a production SLA target).
  Denial-of-service reports against the demo box specifically are not
  actionable; please still report application-layer DoS vectors (e.g. an
  unbounded-loop or resource-exhaustion bug in the code itself).
- Vulnerabilities in third-party dependencies or model weights that are
  already publicly disclosed upstream (report those to the upstream project;
  a note here that we're tracking the CVE is still welcome).
- Social engineering, physical security, or attacks requiring prior
  compromise of a maintainer's credentials.

## Known, Already-Documented Security Posture

To avoid duplicate reports, these are already measured and tracked — see
[CHANGELOG.md](CHANGELOG.md) and the README's Known Limitations section for
current status:
- Prompt injection: 64/64 recall on a 109-case adversarial corpus, all OWASP
  LLM Top 10 (2025) categories addressed.
- Tenant isolation: enforced independently at all four data layers (Qdrant,
  BM25, Redis, MongoDB).
- Secrets: production secrets live in AWS SSM Parameter Store as
  SecureStrings, never committed; `detect-secrets` gates every push.
