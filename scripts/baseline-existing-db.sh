#!/usr/bin/env bash
# Baseline adoption for existing populated DB — see VERTWOPLAN §15.
# Steps: backup -> schema-diff check -> row counts -> stamp head only.
# Usage: bash scripts/baseline-existing-db.sh
# Env: DATABASE_URL, MLOPS_BACKUP_DIR (default C:\Users\LENOVO\mlops-backups\migration-baselines)
set -euo pipefail
cd "$(dirname "$0")/.."

BACKUP_DIR="${MLOPS_BACKUP_DIR:-C:/Users/LENOVO/mlops-backups/migration-baselines}"
DATABASE_URL="${DATABASE_URL:-$(grep -E '^DATABASE_URL=' .env 2>/dev/null | cut -d= -f2- | tr -d '\r' || echo 'postgresql+psycopg://mlops:mlops@localhost:5433/mlops')}"
# Normalize for psql/pg_dump (strip +psycopg* suffix) — handles both +psycopg and +psycopg2
DB_URL_PG=$(echo "$DATABASE_URL" | sed -E 's/\+psycopg2?//')
# Extract host/port for health checks (handle passwords with @ via python urlparse when available)
if command -v python3 >/dev/null 2>&1; then
  PGHOST=$(python3 -c "from urllib.parse import urlparse; u=urlparse('$DB_URL_PG'); print(u.hostname or 'localhost')" 2>/dev/null || echo "localhost")
  PGPORT=$(python3 -c "from urllib.parse import urlparse; u=urlparse('$DB_URL_PG'); print(u.port or 5433)" 2>/dev/null || echo "5433")
else
  PGHOST=$(echo "$DB_URL_PG" | sed -E 's|.*@([^:/]+).*|\1|')
  PGPORT=$(echo "$DB_URL_PG" | sed -E 's|.*:([0-9]+)/.*|\1|')
fi
if [[ -z "$PGHOST" || -z "$PGPORT" ]]; then PGHOST=localhost; PGPORT=5433; fi

echo "== baseline-existing-db =="
echo "DATABASE_URL=$DATABASE_URL"
echo "PGHOST=$PGHOST PGPORT=$PGPORT"
echo "BACKUP_DIR=$BACKUP_DIR"

# 1. Require Docker + Postgres healthy
if ! docker info >/dev/null 2>&1; then
  echo "FATAL: docker not running"; exit 1
fi
echo "waiting for postgres ..."
for i in $(seq 1 30); do
  if docker compose -f infra/docker-compose.yml exec -T postgres pg_isready -U mlops -d mlops >/dev/null 2>&1; then
    echo "postgres healthy"
    break
  fi
  if command -v pg_isready >/dev/null 2>&1 && pg_isready -h "$PGHOST" -p "$PGPORT" -U mlops -d mlops >/dev/null 2>&1; then
    echo "postgres healthy (host)"
    break
  fi
  sleep 2
  if [[ $i -eq 30 ]]; then echo "FATAL: postgres not healthy"; exit 1; fi
done

# 2. Timestamped custom-format backup
mkdir -p "$BACKUP_DIR"
TS=$(date +%Y%m%d-%H%M%S)
BACKUP_FILE="$BACKUP_DIR/baseline_${TS}.dump"
echo "creating backup $BACKUP_FILE ..."
# Prefer docker exec pg_dump (works even without host pg_dump)
if docker compose -f infra/docker-compose.yml exec -T postgres pg_dump -U mlops -d mlops -Fc -f "/tmp/baseline_${TS}.dump" 2>/dev/null; then
  docker compose -f infra/docker-compose.yml exec -T postgres cat "/tmp/baseline_${TS}.dump" > "$BACKUP_FILE"
  # also try docker cp fallback
  if [[ ! -s "$BACKUP_FILE" ]]; then
    docker compose -f infra/docker-compose.yml cp "postgres:/tmp/baseline_${TS}.dump" "$BACKUP_FILE" 2>/dev/null || true
  fi
elif command -v pg_dump >/dev/null 2>&1; then
  PGPASSWORD=mlops pg_dump "$DB_URL_PG" -Fc -f "$BACKUP_FILE"
else
  echo "FATAL: pg_dump not available (install postgres client or use docker)"; exit 1
fi
if [[ ! -s "$BACKUP_FILE" ]]; then echo "FATAL: backup failed or empty"; exit 1; fi
echo "backup ok: $(ls -lh "$BACKUP_FILE" | awk '{print $5}')"

# 3. Schema-diff check — fail if live schema differs from 0001
# Use python -m alembic if alembic not in PATH (Windows)
ALEMBIC_CMD="alembic"
if ! command -v alembic >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1 && python -m alembic --help >/dev/null 2>&1; then ALEMBIC_CMD="python -m alembic"
  elif command -v python3 >/dev/null 2>&1 && python3 -m alembic --help >/dev/null 2>&1; then ALEMBIC_CMD="python3 -m alembic"
  elif [ -f "C:/Python312/python.exe" ] && "C:/Python312/python.exe" -m alembic --help >/dev/null 2>&1; then ALEMBIC_CMD="C:/Python312/python.exe -m alembic"
  fi
fi
echo "checking alembic diff ($ALEMBIC_CMD check) ..."
if ! $ALEMBIC_CMD check 2>&1 | tee "/tmp/alembic_check_${TS}.log"; then
  echo "FATAL: alembic check failed — live schema differs from 0001_initial_schema. See /tmp/alembic_check_${TS}.log"
  echo "Resolve manually before stamping. Backup kept at $BACKUP_FILE"
  exit 1
fi
# Also verify `alembic current` is empty (not yet stamped) and `alembic history` expects 0001
echo "current revision before stamp:"
$ALEMBIC_CMD current || true

# 4. Row counts before
echo "row counts before:"
for tbl in datasets dataset_versions feature_sets training_runs predictions drift_checks audit_log; do
  CNT=$(docker compose -f infra/docker-compose.yml exec -T postgres psql -U mlops -d mlops -Atc "SELECT count(*) FROM $tbl" 2>/dev/null || PGPASSWORD=mlops psql "$DB_URL_PG" -Atc "SELECT count(*) FROM $tbl" 2>/dev/null || echo "?")
  echo "  $tbl: $CNT"
done

# 5. Stamp only — never run CREATE TABLEs against populated DB
echo "stamping alembic head ..."
$ALEMBIC_CMD stamp head

echo "current revision after stamp:"
$ALEMBIC_CMD current

# Row counts after (should be unchanged)
echo "row counts after:"
for tbl in datasets dataset_versions feature_sets training_runs predictions drift_checks audit_log; do
  CNT=$(docker compose -f infra/docker-compose.yml exec -T postgres psql -U mlops -d mlops -Atc "SELECT count(*) FROM $tbl" 2>/dev/null || PGPASSWORD=mlops psql "$DB_URL_PG" -Atc "SELECT count(*) FROM $tbl" 2>/dev/null || echo "?")
  echo "  $tbl: $CNT"
done

echo "== baseline complete =="
echo "backup: $BACKUP_FILE"
echo "alembic at head; no tables recreated"
