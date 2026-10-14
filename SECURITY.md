# Security Policy

## Supported versions

This is a research codebase. Only the latest `main` and the most recent
tagged release receive security considerations.

| Version | Supported          |
|---------|--------------------|
| `main`  | ✅                 |
| `v0.2.x`| ✅                 |
| `< v0.2`| ❌                 |

## Reporting a vulnerability

**Please do NOT open a public GitHub issue for security vulnerabilities.**

Instead, email **kaphlebhaskar012345@gmail.com** with:

1. A description of the issue and its potential impact.
2. Reproduction steps (minimal, if possible).
3. Affected commit SHA or version tag.
4. Any suggested mitigation.

You will receive an acknowledgement within 72 hours and a coordinated
disclosure timeline once the issue is triaged.

## Scope

In scope:
- Vulnerabilities in the package code under `src/nepse_nlp/`.
- Insecure defaults in the FastAPI service (`api/main.py`) or the
  Dockerfile that could lead to RCE, secret leakage, or SSRF.
- CI/CD misconfigurations that leak secrets or allow supply-chain attacks.

Out of scope:
- The simulated Jev backend is a **research stand-in**, not a security
  component — substituting it for the real API is the user's responsibility.
- Findings in optional, un-pinned third-party dependencies (report to upstream).
- Self-induced data exposure from committing secrets — see the hardening
  notes below.

## Hardening notes for deployments

- The `X-Auth` token for any remote command endpoint and the GitHub `ghp_…`
  token must be rotated if exposed. Never commit `.env` or tokens.
- The FastAPI service enables permissive CORS by default for demos — tighten
  `api/main.py` (`cors_origins`) before any public deployment.
- The Streamlit dashboard and FastAPI service are designed for research, not
  production traffic; place behind an auth layer if exposed.
