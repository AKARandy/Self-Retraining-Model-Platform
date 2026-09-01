# Self-Retraining Model Platform
### an MLOps platform

A platform that runs one closed loop end to end:
**live traffic → drift detection → automatic retraining → conditional promotion → served, zero redeploy.**

One FastAPI app (modular monolith) on top of real infrastructure — Postgres, MinIO, MLflow, and Kubernetes
(minikube) running Argo Workflows — plus a React dashboard. It runs entirely in containers on a local
Kubernetes cluster, and the architecture maps 1:1 onto managed cloud services.

**Test case: house prices** (Kaggle House Prices, tabular regression). The dataset is a stand-in —
the pipeline is dataset-agnostic, and §4.1 covers swapping it for any case.

*Keywords: MLOps · Argo Workflows · drift detection · Optuna · MLflow · DVC · FastAPI · Kubernetes.*

---

## 1. What is it

```
Windows host
├─ Docker Desktop VM (WSL2, capped 8 GB via %UserProfile%\.wslconfig)
│   ├─ docker compose (infra/): postgres:16 (host port 5433!) · minio (9000/9001) · mlops-mlflow (5000)
│   └─ minikube cluster (docker driver, 4 CPU / 6 GB)
│       └─ namespace "argo"
│           ├─ WorkflowTemplate train-pipeline:
│           │   ingest → validate → feature-engineer (Featuretools)
│           │   → train (Optuna HPO, nested MLflow runs) ∥ train-nn (PyTorch MLP)
│           │   → evaluate (conditional promote gate) → shap → promote
│           └─ CronWorkflow batch-score (hourly batch scoring)
├─ FastAPI app :8000  (app/data · app/training · app/registry · app/serving · app/monitoring · app/core)
└─ Vite + React dashboard :5173 (polls every 5 s)
```

**The pieces and their jobs**

| Piece | Job |
|---|---|
| `app/data` | dataset uploads → DVC-versioned into MinIO, column stats into Postgres |
| `app/training` | submits the Argo workflow via REST, syncs status into Postgres |
| `app/registry` | wraps MLflow registry; promotion endpoint; model cards; audit log |
| `app/serving` | serves the Production model; rebuilds the engineered feature vector from one raw row via a "recipe" artifact; logs every prediction |
| `app/monitoring` | z-score drift check over the live-prediction window; on breach **auto-submits retraining** |
| MinIO buckets | `dvc-store` (dataset blobs) · `mlflow-artifacts` (models) · `pipeline-artifacts` (parquet/recipes/plots) |
| Postgres | catalog: datasets, feature sets, training runs, predictions, drift checks, audit log (+ separate `mlflow` DB) |

**Design decisions you should know**

- Only the *API layer* is a monolith. Postgres/MinIO/MLflow/Argo stay real, separate services.
- Writes are gated by **one API key** (`Authorization: Bearer <API_KEY>` from `.env`); every write attempt —
  including rejected 401s — lands in the `audit_log` table.
- Drift check is **stats-based** (z-test vs. reference column stats) rather than Evidently; the drift trigger
  runs **host-side** because pod→host:8000 needs a Windows Firewall admin rule. The batch-score CronWorkflow
  shows the in-cluster pattern.
- **One consolidated image** (`mlops-pipeline:dev`) runs all DAG steps and batch scoring — keeps storage low,
  no GPU torch (CPU-only wheel).
- MLflow is pinned to **2.22.x** (server image + client) because the promote flow uses model *stages*
  (deprecated in 3.x, removed later). Migrating to aliases is a documented stretch goal.
- Postgres is on host port **5433** — port 5432 belongs to a native PostgreSQL 17 Windows service.

---

## 2. How to run

### 2a. One-time prerequisites (a new machine)

1. **Docker Desktop** (WSL2 backend) — install and start it.
2. **Python 3.12** + `pip install --user -r requirements.txt`
   (installs fastapi, sqlalchemy, psycopg, dvc[s3], mlflow==2.22.1, scikit-learn, torch-CPU, httpx, …).
3. **Node 18+** (for the dashboard).
4. **minikube** → put `minikube-windows-amd64.exe` in `C:\Users\<you>\bin`, add that dir to PATH.
5. **kubectl** (Docker Desktop bundles one) and **argo CLI**
   (`argo-windows-amd64.exe.gz` from Argo releases → gunzip → `argo.exe` in the same bin dir).
6. Create `%UserProfile%\.wslconfig`:
   ```ini
   [wsl2]
   memory=8GB
   processors=6
   swap=4GB
   ```
   then `wsl --shutdown` once. (16 GB host machines should not give the VM more than 8 GB.)
7. `cd` into the repo, create `.env`:
   ```bash
   printf 'API_KEY=%s\n' "$(openssl rand -hex 24)" > .env
   cat >> .env <<'EOF'
   DATABASE_URL=postgresql+psycopg://mlops:mlops@localhost:5433/mlops
   MINIO_ENDPOINT=localhost:9000
   MLFLOW_TRACKING_URI=http://localhost:5000
   ARGO_URL=https://localhost:2746
   EOF
   ```

### 2b. Bring-up — one command

```bash
bash scripts/up.sh
```

That single idempotent script does everything: starts Docker Desktop if needed, builds/starts the
compose stack (including the **containerized API**), starts minikube, installs Argo on fresh clusters,
applies the workflow templates, loads the pipeline image into the node (with the anti-stale-tag dance),
waits for the control plane, starts the port-forward on `0.0.0.0`, and registers the seed dataset if none exists.

> Pods reach host services via `hostAliases` baked into the workflow templates (Docker Desktop gateway
> `192.168.65.254`) — no CoreDNS patch needed. If you ever change the Docker network and workflows fail
> with `Failed to resolve 'host.minikube.internal'`, update that IP in `infra/k8s/*.yaml`.

Manual equivalent (if you want to run pieces by hand):

```bash
docker compose -f infra/docker-compose.yml up -d      # postgres/minio/mlflow + api container
minikube start
kubectl apply --server-side -n argo -f https://github.com/argoproj/argo-workflows/releases/latest/download/quick-start-minimal.yaml
kubectl apply -f infra/k8s/train-workflow-template.yaml -f infra/k8s/batch-score-cron.yaml
minikube image load mlops-pipeline:dev
kubectl -n argo port-forward --address 0.0.0.0 svc/argo-server 2746:2746 &
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 &   # native dev mode (optional)
cd ui && npm install && npm run dev                    # http://localhost:5173
```

**Verify before using:** `curl localhost:8000/health` → `{"status":"ok"}` · `kubectl get nodes` → Ready ·
Argo UI loads · `curl localhost:5000/health` → OK.

### 2c. First data + first model (fresh install)

```bash
KEY=$(grep '^API_KEY=' .env | cut -d= -f2)
# register the dataset (creates dataset id 1, version 1; DVC-pushes into MinIO)
mkdir -p data/raw/inbox && cp <path-to>/train.csv data/raw/inbox/kaggle_train.csv
curl -X POST localhost:8000/datasets -H "Authorization: Bearer $KEY" \
  -F file=@data/raw/inbox/kaggle_train.csv -F name=house-prices -F target_column=SalePrice
# train (15 Optuna trials ≈ 20-25 min end to end)
curl -X POST localhost:8000/train-runs -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" -d '{"dataset_id":1,"n_trials":15}'
```

The first completed run registers `house-price-sk` (sklearn) and `house-price-nn` (PyTorch) and promotes
the sklearn model to Production (no incumbent exists yet, so the gate passes).

---

## 3. How to use

**Daily driver = the dashboard** (`localhost:5173`): paste the API key once (stored in localStorage),
then watch dataset versions, live training status (auto-polling), registry stages, and drift checks.
Buttons let you submit a run or fire a drift check.

**API walkthrough** (`localhost:8000/docs` has all of it):

| Flow | Command |
|---|---|
| Register data version | `POST /datasets` (multipart) → returns version + DVC md5 |
| List versions / download a version | `GET /datasets/{id}/versions` · `GET /datasets/{id}/versions/{v}/content` |
| Launch training | `POST /train-runs {"dataset_id":1,"dataset_version":2,"n_trials":15}` |
| Poll a run | `GET /train-runs/{id}` (syncs Argo phase into Postgres; includes Argo UI link) |
| Promote | `POST /registry/promote {"name":"house-price-sk","version":N}` — audited |
| Current production model | `GET /registry/models/house-price-sk/current` |
| Predict | `POST /predict {"features":{...raw house columns...},"model":"auto"}` |
| Model card (metrics + params + SHAP) | `GET /registry/models/house-price-sk/versions/N/card` |
| Drift check (auto-retrains on breach) | `POST /monitoring/check-drift` |
| Drift history | `GET /monitoring/drift-checks` |
| Batch score (100 rows, hourly cron) | `argo submit --from cronworkflow/batch-score -n argo -k` |

**The demo everyone remembers:** `bash scripts/demo_drift.sh` — replays 40 drifted predictions through
`/predict`, then triggers the check. Output shows z-scores per feature (e.g. `GrLivArea z=1.6`), and a
retraining workflow appears in the Argo UI with **no human submit**. The promotion gate then honestly
compares the candidate against Production and refuses it if it's worse (reason is recorded).

**Where to look at what:** MLflow UI = trials/runs/models (nested Optuna runs under experiment
`house-prices`); MinIO console = the three buckets; `psql` (host 5433, user `mlops`, db `mlops`) =
`predictions`, `drift_checks`, `audit_log`, `feature_sets`, `training_runs`.

**Gotchas worth knowing before they bite**

- Missing/invalid key → 401, and the rejection is written to `audit_log`.
- `dvc push` can say *"Everything is up to date"* even when MinIO lost objects (remote index).
  Verify with `MC`/boto3 object counts; re-push from `.dvc/cache` if counts are 0.
- Predictions feed drift monitoring — don't delete the `predictions` table between demos.
- Content served for a dataset version is remote-first (MinIO), host `.dvc/cache` fallback.

---

## 4. How to edit / expand

### Repo map

```
app/
  main.py                 FastAPI app factory — include new routers here
  core/                   config (pydantic-settings, .env) · db · models (all tables) · audit · security
  data/                   dataset upload/versions + DVC/MinIO IO (dvc_io.py: add, push, read, artifacts)
  training/               Argo REST submit + status sync (service.py) + routes
  registry/               MLflow registry wrapper + promote + audit + card renderer (card.py)
  serving/                featurizer.py (recipe → feature vector) + model cache + routes
  monitoring/             drift service + routes + host loop (loop.py)
pipeline/                 THE TRAINING CODE — runs inside containers on the cluster
  common.py               container config: API/MLflow/MinIO/DB via host.minikube.internal
  ingest.py validate.py feature_engineer.py train.py train_nn.py evaluate.py shap_step.py promote.py
  batch_score.py          hourly batch scoring
infra/
  docker-compose.yml      postgres(5433)/minio/mlflow + api container (mlops-api:dev)
  k8s/train-workflow-template.yaml    the DAG (8 templates; hostAliases via spec.podSpecPatch)
  k8s/batch-score-cron.yaml
docker/                   api.Dockerfile + api-entrypoint.sh (API container)
                          pipeline.Dockerfile + requirements-training.txt (consolidated cluster image)
scripts/                  up.sh (one-command bring-up) · down.sh (full stop) · demo_drift.sh · make_drift.py
ui/                       React dashboard (App.jsx = all sections; 5 s polling via fetch)
```

### Adding/changing API endpoints

Edit/create a module under `app/<domain>/` with a router, `include_router` it in `app/main.py`.
Write gating = add `dependencies=[Depends(require_api_key)]` on the route and an
`record(db, action, resource, allowed=True, ...)` on success. Tables live centrally in
`app/core/models.py` (schema changes = `create_all()` adds new tables/columns only — no migrations;
if you alter an existing column, drop the table or add Alembic).

### Changing the training pipeline (the common case)

1. Edit the relevant step in `pipeline/*.py` (plain Python; steps talk via MinIO artifacts under
   `houses/<workflow-name>/…` and constants in `pipeline/common.py`).
2. Rebuild + reload the image (or just rerun `scripts/up.sh`, which does it):
   ```bash
   docker build -f docker/pipeline.Dockerfile -t mlops-pipeline:dev .
   minikube ssh "docker ps -aq --filter ancestor=mlops-pipeline:dev --filter status=exited | xargs -r docker rm -f; docker rmi -f mlops-pipeline:dev" 
   minikube image load mlops-pipeline:dev
   ```
   ⚠️ **Never skip the `rmi`** — `minikube image load` does not reliably overwrite an existing tag in the
   node; pods will run stale code or hit `ErrImageNeverPull`. If a pod gets stuck anyway: delete the pod
   (Argo recreates it) — the workflow controller re-runs the node.
3. Re-submit a run. Shakeout pattern: `n_trials: 2` first (~6 min), then 15.

### Adding a DAG step

Add the template to `infra/k8s/train-workflow-template.yaml` (copy an existing step; env is shared via the
`&stepenv` YAML anchor), wire it into the `dag.tasks` list, `kubectl apply`, submit. Keep steps
communicating through artifacts (`common.put_artifact/get_json`) — no shared filesystem.

### Feeding a new model into serving

Train steps register via `mlflow.*.log_model(..., registered_model_name=...)`. Serving loads by
*registered flavor*: sklearn models through `mlflow.sklearn`, PyTorch through `mlflow.pytorch`
(see `app/serving/service.py`; add a branch for new flavors). The model must accept the recipe-built
feature frame — use the same engineered schema.

### The serving recipe (how `/predict` matches training)

`feature_engineer.py` emits `recipe.json` (transform columns + operands, one-hot levels, neighborhood
aggregates, medians, final column list) to `pipeline-artifacts://houses/recipe/latest.json`.
`app/serving/featurizer.py` rebuilds the exact vector from raw input. If you change feature
engineering, the recipe regenerates automatically on the next run — but note the serving cache
refreshes every 60 s (TTL in `featurizer.py`).

---

## 4.1 How to change dataset / case

The pipeline is written for **one flat CSV with an id column, one target, and one grouping column**.
House-prices specifics are deliberately concentrated in a few places:

| What | Where | House-prices value |
|---|---|---|
| Target | `pipeline/common.py` → `TARGET` (overridable via env/param `target`) | `SalePrice` |
| Id column | `pipeline/feature_engineer.py` → `ID_COL` | `Id` |
| Grouping column (the Featuretools child entity) | same file, `build_entityset()` | `Neighborhood` |
| Columns aggregated per group | same file → `AGG_COLUMNS` | `LotArea, OverallQual, YearBuilt, GrLivArea` |
| Primitives | same file → `TRANSFORM_PRIMITIVES`, `AGG_PRIMITIVES` | add/multiply transforms + group aggs |
| Optuna search space | `pipeline/train.py` → `sample_params()` / `build_model()` | HistGB + RandomForest |
| NN shape | `pipeline/train_nn.py` → `MLP` class | 128/64/32 MLP |
| Drift simulation columns | `scripts/make_drift.py` | GrLivArea/YearBuilt/OverallQual/LotArea |

**Recipe for a new case (e.g. credit default, churn):**

1. Edit the five anchors above for your schema (keep `max_depth=1` unless you understand the depth/agg note below).
2. Upload as a **new dataset name** — `curl -F name=credit-default …`. Versioning is per-name;
   the feature set provenance rows link to the dataset version automatically.
3. Pass the target at submit time (WorkflowTemplate param `target`, or bake it into `common.py`).
4. Update `app/serving/service.py` defaults (`default_model`) and the registry model names used in
   `evaluate.py`/`promote.py` if you rename models (`house-price-sk` → your name).
5. Re-run; the recipe, feature set rows, and cards all regenerate. Serving needs no code change
   (it's recipe-driven) unless your case needs new feature *kinds*.

**Honest limitations:** (a) at `max_depth=1` Featuretools emits transform + direct features — the
neighborhood-aggregation branch exists but produces no agg columns at this depth (depth-2 is the
documented next step); (b) categoricals are one-hot with NaN dummies; (c) regression metrics
(RMSE/MAE/R²) are hardcoded in `train.py`/`evaluate.py` — classification means swapping those and
the SHAP step's explainer.

---

## 5. How to stop

**One command:** `bash scripts/down.sh` — kills the API/dashboard processes, removes the compose
containers (**volumes kept**), freezes minikube, and stops the port-forward. `scripts/up.sh` reverses it.

**Manual equivalent:**

```bash
# stop the app + UI (Ctrl+C on uvicorn / npm run dev, kill the port-forward), then:
minikube stop                                        # freezes the cluster (fast restart later)
docker compose -f infra/docker-compose.yml stop      # stops postgres/minio/mlflow/api (data kept)
```

**Full stop (frees all Docker/WSL resources):**

```bash
docker compose -f infra/docker-compose.yml down      # removes containers; volumes kept
minikube stop
# then quit Docker Desktop from the tray
```

**Destroying data (know the difference):**

- `docker compose … down -v` — deletes Postgres catalog + MinIO objects (datasets, models, artifacts).
  Don't without a backup.
- `minikube delete` — deletes the cluster (Argo install, workflow history). Rebuildable from §2b in ~5 min.
- Reset MinIO/Postgres from backups: see `C:\Users\LENOVO\mlops-backups` (volumes as `.tgz`,
  images as `images.tar`; restore = `docker load -i images.tar`, create volumes, `tar xzf` into them).

**Storage notes**

- Real footprint ≈ 11 GB inside Docker's `docker_data.vhdx`. That file grows but never shrinks by itself;
  WSL2 can't compact it without admin tools. If C: creeps down after heavy rebuilds: prune
  (`docker image prune -af`, and inside the node `minikube ssh "docker system prune -af"`), then compact
  with an elevated `diskpart` (`select vdisk file="…docker_data.vhdx"` → `attach vdisk readonly` →
  `compact vdisk` → `detach vdisk`) with Docker fully quit. Expect modest reclaim — most of the file is live data.
- `dvc` cache lives in `.dvc/cache` (host); large deletes there are safe — objects re-push from it.

---

## Verified end-to-end (git history = evidence)

Two dataset versions registered with DVC md5-exact round-trips · 15-trial Optuna studies as nested MLflow
runs · 8-node Argo DAG `Succeeded` (twice — including on a fully rebuilt cluster) · sklearn + PyTorch models
registered · SHAP artifacts · conditional promotion gate refusing a worse candidate · drift breach (z≈1.6)
auto-triggering retraining · zero-redeploy serving swap · 100-row batch scoring · 200/401 auth pair with
audit rows · live dashboard. Honest caveat kept on purpose: the promotion gate once refused a candidate —
the earlier flashier metrics were target leakage (caught via SHAP), fixed, and the clean model is what serves.
