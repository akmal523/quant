# Security Policy

## Reporting a vulnerability

Report vulnerabilities privately via
[GitHub Security Advisories](https://github.com/akmal523/quant/security/advisories/new).
Do **not** open a public issue for security problems.

Include:

- Affected version / commit
- Reproduction steps
- Impact (data exposure, arbitrary code execution, etc.)

We aim to acknowledge reports within 48 hours and to provide a remediation plan
within 7 days.

## Scope

This project reads market data and broker exports and stores them locally in
DuckDB. It does not run a network service. Highest-risk areas:

- Untrusted input parsing (`portfolio.csv`, broker exports, SEC/news HTML)
- Credentials via `.env` (never commit `.env`; it is gitignored)

## Supported versions

Only the latest release line receives security fixes.
