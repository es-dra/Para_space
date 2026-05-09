# Security

> Extends `~/.claude/rules/common/security.md` and `~/.claude/rules/python/security.md`.

---

## Credential Management

- **No hardcoded API keys, passwords, or tokens** in any file under version control
- Use environment variables or `.env` files (`.env` must be in `.gitignore`)
- The project currently has NO API-dependent code — this is a local ML research project

---

## Known Issue

`~/.claude/settings.json` currently contains hardcoded API keys in plain text. This is a **system-level** issue in the Claude Code configuration, not in this project's code. It should be fixed by using environment variable references instead of literal keys.

---

## Research Data

- All image datasets (Set5, Set14, BSD100, DIV2K) are public academic benchmarks
- No PII, no user data, no confidentiality concerns
- Results (NPZ trajectories, JSON summaries) contain only model weights and metrics — no sensitive content

---

## Static Analysis

```bash
bandit -r src/          # Run before committing library changes
pip-audit               # Check dependencies for known vulnerabilities
```
