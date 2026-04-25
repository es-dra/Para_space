---
paths:
  - "**/*.py"
  - "**/*.pyi"
---

# Coding Standards

> Extends `~/.claude/rules/common/coding-style.md` and `~/.claude/rules/python/coding-style.md`.
> This file adapts global coding rules to the research-code context.

---

## 1. Karpathy Principles (Pre-Coding Mandate)

Apply these **before every implementation task**:

### 1.1 Think Before Coding
**Don't assume. Don't hide confusion. Surface tradeoffs.**

- State assumptions explicitly. Ask when uncertain.
- When multiple interpretations exist, present them — don't silently pick one.
- Proactively offer simpler alternatives when they exist.
- Stop and ask when something is unclear.

### 1.2 Simplicity First
**Minimum code that solves the problem. Nothing speculative.**

- Ask: "Is this overly complex?"
- No features that weren't requested.
- No abstractions for single-use cases.
- No error handling for scenarios that cannot happen.
- If 200 lines can be 50, rewrite.

### 1.3 Surgical Changes
**Touch only what you must. Clean up only your own mess.**

- Only change what's necessary.
- Don't "improve" adjacent code, comments, or formatting while doing something else.
- Don't refactor things that aren't broken.
- If your changes leave orphaned code/functions, delete them.
- Don't delete pre-existing dead code (unless asked).

### 1.4 Goal-Driven Execution
**Define success criteria. Loop until verified.**

- "Fix the bug" → "Write a test reproducing the bug first, then fix."
- "Add feature X" → "Define the acceptance test first, then implement."
- Multi-step tasks: state a short plan with verification at each step.

---

## 2. Python Standards

- **PEP 8** conventions throughout
- **Type annotations** on all function signatures in library code
- **Immutability**: return new dicts/tensors; never mutate inputs in-place
- Files ≤ 500 lines; extract helpers when approaching that limit
- Error at system boundaries (file I/O, CLI args, external data). Trust internal contracts.

---

## 3. Library vs Experiment Code

| Aspect | Library (`src/`) | Experiment Scripts |
|--------|-------------------|--------------------|
| Logging | `logging` module | `print()` is fine |
| Abstractions | DRY, reusable | Simplicity > elegance; duplication acceptable |
| Error handling | Explicit, at boundaries | Crash loudly is OK |
| Config | Typed, validated | Hardcoded values matching config are OK |
| Type annotations | Required | Optional |

---

## 4. Naming Conventions

- Files: `snake_case.py` (Python), `PascalCase` only for class-named files if following existing pattern
- Result files: `{transform}_{method}_{model}_{image}_seed{seed}.{ext}`
- Experiment directories: `Phase{N}_{Transform}_{Method}/` (e.g., `Phase1_Scale_Independent/`)
