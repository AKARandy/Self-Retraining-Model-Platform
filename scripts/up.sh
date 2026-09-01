#!/usr/bin/env bash
# One-command bring-up: services, cluster, Argo, pipeline image, port-forward, API.
# Idempotent — safe to run repeatedly. On first run it builds the API image (~5-10 min).
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n== %s ==\n' "$1"; }

step "0/7 docker engine"
if ! docker info >/dev/null 2>&1; then
  echo "starting Docker Desktop..."
  powershell -NoProfile -Command "Start-Process 'C:\Program Files\Docker\Docker\Docker Desktop.exe'"
  for i in $(seq 1 60); do
    if docker info >/dev/null 2>&1; then echo "engine up"; break; fi
    sleep 5
  done
  docker info >/dev/null 2>&1 || { echo "FATAL: docker engine did not start"; exit 1; }
else
  echo "engine already up"
fi

step "1/7 services (postgres, minio, mlflow, api)"
docker compose -f infra/docker-compose.yml up -d --build 2>&1 | grep -vE "^ |^$" | tail -4 || true
for i in $(seq 1 60); do
  if curl -s -m 3 http://localhost:8000/health >/dev/null 2>&1; then echo "api healthy"; break; fi
  sleep 3
done
curl -s -m 5 http://localhost:8000/health || { echo "FATAL: api did not come up"; exit 1; }

step "2/7 cluster"
if [ "$(minikube status --format '{{.Host}}' 2>/dev/null)" = "Running" ]; then
  echo "minikube already running"
else
  minikube start
fi

step "3/7 argo workflows"
if ! kubectl get namespace argo >/dev/null 2>&1; then kubectl create namespace argo; fi
if ! kubectl -n argo get deploy argo-server >/dev/null 2>&1; then
  echo "installing argo (first time on this cluster)..."
  kubectl apply --server-side -n argo -f https://github.com/argoproj/argo-workflows/releases/latest/download/quick-start-minimal.yaml
fi
kubectl apply -f infra/k8s/train-workflow-template.yaml
kubectl apply -f infra/k8s/batch-score-cron.yaml

step "4/7 pipeline image into node"
# rmi first: 'minikube image load' does not reliably overwrite an existing tag
minikube ssh "docker ps -aq --filter ancestor=mlops-pipeline:dev --filter status=exited | xargs -r docker rm -f; docker rmi -f mlops-pipeline:dev" >/dev/null 2>&1 || true
minikube image load mlops-pipeline:dev
minikube ssh "docker run --rm mlops-pipeline:dev python -c 'print(\"node image ok\")'" 2>&1 | tail -1

step "5/7 waiting for argo control plane"
kubectl -n argo rollout status deploy/argo-server --timeout=420s
kubectl -n argo rollout status deploy/workflow-controller --timeout=120s

step "6/7 argo port-forward (0.0.0.0 so containers/clusters can reach it)"
oldpid=$(netstat -ano | grep ":2746" | grep LISTENING | awk '{print $NF}' | head -1 || true)
if [ -n "${oldpid:-}" ]; then taskkill //F //PID "$oldpid" >/dev/null 2>&1 || true; fi
nohup kubectl -n argo port-forward --address 0.0.0.0 svc/argo-server 2746:2746 >/dev/null 2>&1 &
sleep 5
code=$(curl -sk -m 10 https://localhost:2746/api/v1/workflows/argo -o /dev/null -w '%{http_code}')
[ "$code" = "200" ] || { echo "FATAL: argo api not reachable (got $code)"; exit 1; }
echo "argo api: 200"

step "7/7 seed check"
if [ "$(curl -s -m 5 http://localhost:8000/datasets)" = "[]" ] && [ -f data/raw/inbox/kaggle_train.csv ]; then
  KEY=$(grep '^API_KEY=' .env | cut -d= -f2)
  echo "registering seed dataset..."
  curl -s -m 180 -X POST localhost:8000/datasets -H "Authorization: Bearer $KEY" \
    -F "file=@data/raw/inbox/kaggle_train.csv" -F "name=house-prices" -F "target_column=SalePrice" | head -c 200
  echo
else
  echo "dataset already registered (or no inbox csv) — skipping"
fi

KEY=$(grep '^API_KEY=' .env | cut -d= -f2)
cat <<'EOS'

================ UP ================
API        http://localhost:8000/docs
Dashboard  http://localhost:5173        (cd ui && npm run dev, if not running)
Argo UI    https://localhost:2746       (self-signed cert)
MLflow     http://localhost:5000
MinIO      http://localhost:9001        (minioadmin / minioadmin)
Train:     curl -X POST localhost:8000/train-runs -H "Authorization: Bearer $KEY" \
             -H "Content-Type: application/json" -d '{"dataset_id":1,"n_trials":15}'
Drift demo: bash scripts/demo_drift.sh
EOS
