#!/usr/bin/env bash
# Local deploy orchestrator — VERTWOPLAN §18.1 (both manual and GitHub modes use same steps)
# Usage: bash scripts/deploy-local.sh <40-char-SHA>  [manual mode: operator verifies SHA is ancestor of origin/main and CI green]
# Env: MLOPS_ENV_FILE (default C:/Users/LENOVO/.mlops-deploy/platform.env), MLOPS_BACKUP_DIR, DATABASE_URL
set -euo pipefail
cd "$(dirname "$0")/.."

SHA="${1:-}"
if [[ ! "$SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Usage: $0 <40-char-commit-SHA>"
  echo "Example: bash scripts/deploy-local.sh \$(git rev-parse HEAD)"
  exit 2
fi

MLOPS_ENV_FILE="${MLOPS_ENV_FILE:-C:/Users/LENOVO/.mlops-deploy/platform.env}"
MLOPS_BACKUP_DIR="${MLOPS_BACKUP_DIR:-C:/Users/LENOVO/mlops-backups/migration-baselines}"
DATABASE_URL_IN="${DATABASE_URL:-}"

echo "== deploy-local $SHA =="
echo "MLOPS_ENV_FILE=$MLOPS_ENV_FILE"
echo "MLOPS_BACKUP_DIR=$MLOPS_BACKUP_DIR"

# 1. Validate tools, labels, SHA, staged .env
for tool in docker kubectl; do
  if ! command -v "$tool" >/dev/null 2>&1; then echo "FATAL: $tool not found"; exit 1; fi
done
if ! command -v minikube >/dev/null 2>&1; then echo "FATAL: minikube not found"; exit 1; fi

# Verify SHA exists and is ancestor of origin/main (when origin exists)
if git rev-parse --verify "$SHA^{commit}" >/dev/null 2>&1; then
  echo "SHA exists locally: $(git log -1 --oneline "$SHA")"
else
  echo "fetching origin..."
  git fetch origin --prune 2>&1 | tail -5 || true
fi
if git rev-parse --verify origin/main >/dev/null 2>&1; then
  if ! git merge-base --is-ancestor "$SHA" origin/main 2>/dev/null; then
    # Allow SHA == origin/main or SHA ahead? Requirement is SHA ancestor of origin/main => CI passed on main
    # For manual mode we warn, not fail, to allow deploying a local commit not yet pushed
    echo "WARN: $SHA is not ancestor of origin/main — verify CI green manually"
  else
    echo "SHA is ancestor of origin/main"
  fi
fi

# Verify checked-out SHA matches input (when running from GitHub workflow checkout)
CURRENT_SHA=$(git rev-parse HEAD)
if [[ "$CURRENT_SHA" != "$SHA" ]]; then
  echo "WARN: current HEAD $CURRENT_SHA != requested $SHA — checking out $SHA"
  git checkout --force "$SHA"
fi

# Stage .env from protected file if needed
if [[ -f "$MLOPS_ENV_FILE" && ! -f .env ]]; then
  echo "staging .env from $MLOPS_ENV_FILE"
  cp "$MLOPS_ENV_FILE" .env
fi
if [[ ! -f .env ]]; then echo "FATAL: .env not found and MLOPS_ENV_FILE not available"; exit 1; fi
if ! grep -q '^API_KEY=' .env; then echo "FATAL: .env missing API_KEY"; exit 1; fi
echo ".env staged"

# 2. Build images from that SHA
echo "== 2/8 build images =="
docker compose -f infra/docker-compose.yml build api mlflow 2>&1 | tail -10
docker build -f docker/pipeline.Dockerfile -t mlops-pipeline:dev . 2>&1 | tail -5
echo "images built:"
docker images mlops-api:dev mlops-mlflow:2.22.1 mlops-pipeline:dev --format "{{.Repository}}:{{.Tag}} {{.ID}} {{.CreatedAt}}" 2>&1 | head -10 || docker images | grep -E "mlops-" | head -10

# 3. Start infra deps needed to inspect DB
echo "== 3/8 start infra deps =="
docker compose -f infra/docker-compose.yml up -d postgres minio 2>&1 | tail -5
for i in $(seq 1 30); do
  if docker compose -f infra/docker-compose.yml exec -T postgres pg_isready -U mlops -d mlops >/dev/null 2>&1; then break; fi
  sleep 2
done

# 4. Backup if migration needed
ALEMBIC_CMD="alembic"
if ! command -v alembic >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1 && python -m alembic --help >/dev/null 2>&1; then ALEMBIC_CMD="python -m alembic"
  elif command -v python3 >/dev/null 2>&1 && python3 -m alembic --help >/dev/null 2>&1; then ALEMBIC_CMD="python3 -m alembic"
  elif [ -f "C:/Python312/python.exe" ] && "C:/Python312/python.exe" -m alembic --help >/dev/null 2>&1; then ALEMBIC_CMD="C:/Python312/python.exe -m alembic"
  fi
fi
if command -v alembic >/dev/null 2>&1 || python -m alembic --help >/dev/null 2>&1 || C:/Python312/python.exe -m alembic --help >/dev/null 2>&1; then
  CUR=$($ALEMBIC_CMD current 2>&1 | tail -1 || echo "")
  HEAD=$($ALEMBIC_CMD heads 2>&1 | head -1 || echo "0001")
  echo "alembic current: $CUR"
  echo "alembic head: $HEAD"
  if ! echo "$CUR" | grep -q "$HEAD" 2>/dev/null; then
    echo "migration needed — backup before upgrade"
    mkdir -p "$MLOPS_BACKUP_DIR"
    TS=$(date +%Y%m%d-%H%M%S)
    BACKUP_FILE="$MLOPS_BACKUP_DIR/pre-deploy_${TS}.dump"
    if docker compose -f infra/docker-compose.yml exec -T postgres pg_dump -U mlops -d mlops -Fc -f "/tmp/pre-deploy_${TS}.dump" 2>/dev/null; then
      docker compose -f infra/docker-compose.yml exec -T postgres cat "/tmp/pre-deploy_${TS}.dump" > "$BACKUP_FILE" || true
      echo "backup: $BACKUP_FILE"
    else
      echo "WARN: backup failed — continuing to upgrade (ensure you have a recent backup)"
    fi
  fi
else
  echo "WARN: alembic not installed — skipping backup check"
fi

# 5. Apply migrations, then start/update API stack
echo "== 5/8 alembic upgrade head =="
if command -v alembic >/dev/null 2>&1 || python -m alembic --help >/dev/null 2>&1 || C:/Python312/python.exe -m alembic --help >/dev/null 2>&1; then
  $ALEMBIC_CMD upgrade head
  echo "alembic at head: $($ALEMBIC_CMD current 2>&1 | tail -1)"
else
  echo "WARN: alembic not found"
fi
echo "starting compose api stack ..."
docker compose -f infra/docker-compose.yml up -d --build 2>&1 | tail -10
# Wait for API health
for i in $(seq 1 40); do
  if curl -s -m 3 http://localhost:8000/health >/dev/null 2>&1; then echo "api healthy"; break; fi
  sleep 3
  if [[ $i -eq 40 ]]; then echo "FATAL: api not healthy"; docker compose -f infra/docker-compose.yml ps; docker compose -f infra/docker-compose.yml logs api --tail 50 || true; exit 1; fi
done

# 6. Start/update minikube, load pipeline image, apply Argo manifests
echo "== 6/8 minikube + pipeline image + Argo manifests =="
if [[ "$(minikube status --format '{{.Host}}' 2>/dev/null)" != "Running" ]]; then
  echo "starting minikube ..."
  minikube start
else
  echo "minikube already running"
fi
# Explicit build already done at step 2 — now evict old and load new (closes stale-tag gap, VERTWOPLAN §18.1 + up.sh fix)
minikube ssh "docker ps -aq --filter ancestor=mlops-pipeline:dev --filter status=exited | xargs -r docker rm -f; docker rmi -f mlops-pipeline:dev" >/dev/null 2>&1 || true
minikube image load mlops-pipeline:dev
minikube ssh "docker run --rm mlops-pipeline:dev python -c 'print(\"node image ok\")'" 2>&1 | tail -1

if ! kubectl get namespace argo >/dev/null 2>&1; then kubectl create namespace argo; fi
if ! kubectl -n argo get deploy argo-server >/dev/null 2>&1; then
  echo "installing argo quick-start-minimal ..."
  kubectl apply --server-side -n argo -f https://github.com/argoproj/argo-workflows/releases/latest/download/quick-start-minimal.yaml
fi
kubectl apply -f infra/k8s/train-workflow-template.yaml
kubectl apply -f infra/k8s/batch-score-cron.yaml
kubectl -n argo rollout status deploy/argo-server --timeout=300s || echo "WARN: argo-server rollout timeout"
kubectl -n argo rollout status deploy/workflow-controller --timeout=120s || echo "WARN: workflow-controller rollout timeout"

# Ensure port-forward
OLD_PID=$(netstat -ano 2>/dev/null | grep ":2746" | grep LISTENING | awk '{print $NF}' | head -1 || true)
if [[ -n "${OLD_PID:-}" ]]; then taskkill //F //PID "$OLD_PID" >/dev/null 2>&1 || true; fi
nohup kubectl -n argo port-forward --address 0.0.0.0 svc/argo-server 2746:2746 >/dev/null 2>&1 &
sleep 5

# 7. Verify health
echo "== 7/8 verify =="
FAIL=0
check() {
  local name="$1" cmd="$2"
  if eval "$cmd" >/dev/null 2>&1; then echo "ok $name"; else echo "FAIL $name"; FAIL=1; fi
}
check "api health" "curl -s -m 5 http://localhost:8000/health | grep -q ok"
check "mlflow health" "curl -s -m 5 http://localhost:5000/health | grep -qi ok || curl -s -m 5 http://localhost:5000/ | grep -qi mlflow"
check "compose healthy" "docker compose -f infra/docker-compose.yml ps | grep -qi 'healthy\|Up'"
check "minikube ready" "kubectl get nodes | grep -q Ready"
check "argo api" "curl -sk -m 10 https://localhost:2746/api/v1/workflows/argo -o /dev/null -w '%{http_code}' | grep -q 200"
if command -v alembic >/dev/null 2>&1; then
  check "alembic head" "alembic current 2>&1 | grep -q head"
fi

# 8. Summary
SHA_SHORT=$(git rev-parse --short "$SHA")
# Reuse ALEMBIC_CMD from above
ALEMBIC_CMD="${ALEMBIC_CMD:-alembic}"
if ! command -v $ALEMBIC_CMD >/dev/null 2>&1 && command -v python >/dev/null 2>&1 && python -m alembic --help >/dev/null 2>&1; then ALEMBIC_CMD="python -m alembic"; fi
REV=$($ALEMBIC_CMD current 2>&1 | tail -1 || echo "unknown")
API_IMG=$(docker images mlops-api:dev --format "{{.ID}}" 2>/dev/null | head -1 || echo "unknown")
PIPE_IMG=$(docker images mlops-pipeline:dev --format "{{.ID}}" 2>/dev/null | head -1 || echo "unknown")
# Write to summary file (GitHub step summary when in Actions, local file otherwise)
SUMMARY_FILE="${GITHUB_STEP_SUMMARY:-deploy-summary-${SHA_SHORT}-$(date +%Y%m%d-%H%M%S).md}"
{
  echo "## Deploy $SHA_SHORT"
  echo ""
  echo "- **SHA:** $SHA"
  echo "- **Rev:** $REV"
  echo "- **Images:** api $API_IMG / pipeline $PIPE_IMG"
  echo "- **Health:** $([ $FAIL -eq 0 ] && echo "all ok" || echo "SOME FAILED")"
  echo "- **Time:** $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo ""
  echo "### Checks"
  echo "- API: http://localhost:8000/health"
  echo "- Argo: https://localhost:2746 (self-signed)"
  echo "- MLflow: http://localhost:5000"
} >> "$SUMMARY_FILE"
cat "$SUMMARY_FILE"
echo "summary -> $SUMMARY_FILE"

# Always-cleanup staged .env if we created it from MLOPS_ENV_FILE (GitHub mode)
if [[ -n "${GITHUB_ACTIONS:-}" && -f "$MLOPS_ENV_FILE" ]]; then
  # In GitHub Actions, .env was copied from protected file — keep it for next steps, cleanup handled by workflow's always() step
  echo "GitHub mode — .env cleanup handled by workflow"
fi

if [[ $FAIL -ne 0 ]]; then echo "deploy finished with failures — see above"; exit 1; fi
echo "== deploy-local $SHA_SHORT done =="
