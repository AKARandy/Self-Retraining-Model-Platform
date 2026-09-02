#!/usr/bin/env bash
# Full stop: app processes, compose services (volumes KEPT), cluster frozen, port-forward killed.
# Images and volumes survive; `scripts/up.sh` brings everything back.
cd "$(dirname "$0")/.."

echo "== app processes (api :8000, ui :5173) =="
powershell -NoProfile -ExecutionPolicy Bypass -Command '
foreach ($port in 8000, 5173) {
  Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
    $p = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
    if ($p -and ($p.ProcessName -in "python","node")) { Stop-Process -Id $p.Id -Force; "killed $($p.ProcessName) on $port" }
  }
}' || echo "(nothing running on 8000/5173)"

echo "== compose services (containers removed, volumes kept) =="
docker compose -f infra/docker-compose.yml down 2>&1 | tail -2 || echo "(docker engine not running)"

echo "== cluster =="
minikube stop 2>&1 | tail -1 || echo "(minikube not running)"

echo "== argo port-forward =="
powershell -NoProfile -ExecutionPolicy Bypass -Command '
Get-NetTCPConnection -LocalPort 2746 -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
  $p = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
  if ($p -and $p.ProcessName -eq "kubectl") { Stop-Process -Id $p.Id -Force; "killed kubectl port-forward" }
}' || echo "(no port-forward running)"

echo "Stopped. Volumes, images and cluster state preserved — scripts/up.sh to return."
exit 0
