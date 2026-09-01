#!/usr/bin/env bash
# Closed-loop drift demo:
#  1. replay drifted traffic through /predict (features shifted off the reference)
#  2. trigger the drift check -> it auto-submits a retraining workflow in Argo
#  3. poll the check + the resulting training run
set -euo pipefail
cd "$(dirname "$0")/.."

KEY=$(grep '^API_KEY=' .env | cut -d= -f2)
API=http://localhost:8000

echo "== replaying 40 drifted predictions =="
python - <<'EOF'
import json, sys
import pandas as pd, requests

df = pd.read_csv("data/raw/inbox/kaggle_train.csv").sample(40, random_state=1)
df["GrLivArea"] = (df["GrLivArea"] * 1.6).round(0)      # strong shift
df["LotArea"] = (df["LotArea"] * 1.5).round(0)
df["YearBuilt"] = df["YearBuilt"] + 40
key = open(".env").read().split("API_KEY=")[1].splitlines()[0]
ok = 0
for _, row in df.iterrows():
    r = requests.post("http://localhost:8000/predict",
                      headers={"Authorization": f"Bearer {key}"},
                      json={"features": {k: (None if pd.isna(v) else v) for k, v in row.items()}},
                      timeout=30)
    ok += r.status_code == 200
print(f"predictions: {ok}/40 accepted")
EOF

echo "== triggering drift check =="
curl -s -X POST "$API/monitoring/check-drift" -H "Authorization: Bearer $KEY" | python -m json.tool
