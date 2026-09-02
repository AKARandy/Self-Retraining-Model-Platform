

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_read_routes_open_without_auth(client, db):
    # GETs are open — no auth required (mock MLflow for registry)
    from unittest.mock import patch

    assert client.get("/datasets").status_code == 200
    assert client.get("/train-runs").status_code == 200
    with patch("app.registry.service.MlflowClient") as MockClient:
        MockClient.return_value.search_registered_models.return_value = []
        MockClient.return_value.search_model_versions.return_value = []
        assert client.get("/registry/models").status_code == 200
    assert client.get("/monitoring/drift-checks").status_code == 200
