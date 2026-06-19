# Commit Format Enforcement — Demo

This branch enforces a commit message format at 3 levels.

---

## The Rule

Every commit message must follow this format:

```
type: description
```

**Valid types:** `feature`, `fix`, `docs`, `refactor`, `test`, `chore`

---

## The 3 Files

| File | What it does |
|------|-------------|
| [SKILL.md](./SKILL.md) | Cursor AI reads this and enforces the format when you commit via Cursor |
| [install-hooks.sh](./install-hooks.sh) | Run once after cloning — installs a git hook that blocks bad commits from terminal |
| [.github/workflows/commit-lint.yml](./.github/workflows/commit-lint.yml) | GitHub Actions — checks every push and PR automatically |

---

## Live Examples

### ✅ Accepted commits (on this branch)
```
feature: add commit message format enforcement skill
docs: add DEMO.md with examples and file links
```

### ❌ Rejected commits (blocked by the git hook)
```
updated stuff         → no type prefix
Fixed bug             → wrong format
WIP                   → no type prefix
new feature added     → wrong format
```

When someone tries a bad commit from the terminal, they see:

```
❌ Commit rejected — bad message format.
   Got:      "updated stuff"
   Expected: type: description
   Valid types: feature, fix, docs, refactor, test, chore

   Examples:
     feature: add login page
     fix: resolve null pointer in auth
```

---

## How to Install the Hook (one-time)

```bash
bash install-hooks.sh
```

After that, every terminal commit is checked automatically.
