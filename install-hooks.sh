#!/bin/bash
# Run this once after cloning to install the commit-msg git hook.

HOOK=".git/hooks/commit-msg"

cat > "$HOOK" << 'EOF'
#!/bin/bash
# Enforces: type: description
PATTERN='^(feature|fix|docs|refactor|test|chore): .+'
MSG=$(cat "$1")

if ! echo "$MSG" | grep -qE "$PATTERN"; then
  echo ""
  echo "❌ Commit rejected — bad message format."
  echo "   Got:      \"$MSG\""
  echo "   Expected: type: description"
  echo "   Valid types: feature, fix, docs, refactor, test, chore"
  echo ""
  echo "   Examples:"
  echo "     feature: add login page"
  echo "     fix: resolve null pointer in auth"
  echo ""
  exit 1
fi
EOF

chmod +x "$HOOK"
echo "✅ commit-msg hook installed."
