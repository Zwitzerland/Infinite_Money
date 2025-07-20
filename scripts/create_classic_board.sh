#!/usr/bin/env bash
set -euo pipefail

REPO="Zwitzerland/Infinite_Money"
BOARD_NAME="Sprint Alpha – AlphaQuanta Q-v2 (2025-07-19 → 2025-08-02)"
COLUMNS=("Todo" "In Progress" "Review" "Done")
ISSUES=(1 2 3 4 5 6 7)   # issue numbers already created

api () {
  curl -sSL -H "Authorization: token $GITHUB_TOKEN" \
       -H "Accept: application/vnd.github.inertia-preview+json" \
       -H "Content-Type: application/json" \
       "$@"
}

echo "➜ Creating Classic Project board…"
BOARD_ID=$(api -X POST \
  -d "{\"name\":\"$BOARD_NAME\",\"body\":\"Quantum-hybrid UAT, risk hardening, diffusion alpha, full soak.\"}" \
  "https://api.github.com/repos/$REPO/projects" | jq -r .id)

if [[ "$BOARD_ID" == "null" || -z "$BOARD_ID" ]]; then
  echo "❌ Failed to create board. Check token scope or permissions."; exit 1; fi
echo "✓ Board id = $BOARD_ID"

declare -A COLUMN_ID
for COL in "${COLUMNS[@]}"; do
  ID=$(api -X POST -d "{\"name\":\"$COL\"}" \
       "https://api.github.com/projects/$BOARD_ID/columns" | jq -r .id)
  COLUMN_ID["$COL"]=$ID
  echo "  • Column '$COL' id=$ID"
done

for NUM in "${ISSUES[@]}"; do
  api -X POST \
    -d "{\"content_id\":$(api https://api.github.com/repos/$REPO/issues/$NUM | jq .id),\"content_type\":\"Issue\"}" \
    "https://api.github.com/projects/columns/${COLUMN_ID[Todo]}/cards" > /dev/null
  echo "    → Added issue #$NUM to Todo"
done

echo "🚀 Classic board ready:"
echo "   https://github.com/$REPO/projects/$(api https://api.github.com/repos/$REPO/projects | jq '.[]|select(.name=="'"$BOARD_NAME"'")|.number')"
