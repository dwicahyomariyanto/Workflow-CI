# Workflow-CI

Re-training model Credit Scoring otomatis dengan **MLflow Project** + **GitHub Actions**.

```
Workflow-CI
├── .github/workflows/ci.yml
└── MLProject
    ├── MLProject              # definisi MLflow Project
    ├── conda.yaml
    ├── modelling.py
    ├── credit_preprocessing/  # dataset siap latih
    └── DockerHub.txt          # tautan Docker Hub
```

Alur workflow: checkout → setup Python 3.12.7 → install dependencies → `mlflow run` →
ambil run_id terbaru → simpan artefak ke repo (`artifacts/`) & GitHub Artifacts →
`mlflow models build-docker` → push image ke Docker Hub.

Secrets yang dibutuhkan (Settings → Secrets and variables → Actions):
`DOCKER_HUB_USERNAME`, `DOCKER_HUB_ACCESS_TOKEN`.

Docker image: https://hub.docker.com/r/dwicahyom16/credit-scoring
