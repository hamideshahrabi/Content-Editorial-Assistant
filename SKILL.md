# Commit Message Format Skill

## Rule
Every commit message must follow this exact format:

```
type: description
```

## Valid Types
- `feature` — new functionality
- `fix` — bug fix
- `docs` — documentation changes
- `refactor` — code restructuring without behavior change
- `test` — adding or updating tests
- `chore` — maintenance tasks (dependencies, config, etc.)

## Examples

✅ Valid:
```
feature: add Persian translation caching
fix: resolve session timeout on long translations
docs: update API usage instructions
refactor: simplify translation pipeline logic
```

❌ Invalid:
```
updated stuff
Fixed bug
WIP
new feature added
```

## Enforcement
This rule is enforced at three levels:
1. **Cursor** — this SKILL.md file (AI assistant enforces it when you commit via Cursor)
2. **Git hook** — `.git/hooks/commit-msg` rejects bad messages from the terminal
3. **GitHub Actions** — `.github/workflows/commit-lint.yml` checks every push/PR
# Commit format is enforced here
