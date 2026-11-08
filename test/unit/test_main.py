from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint() -> None:
    response = TestClient(app).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unknown_endpoint_is_not_healthy() -> None:
    response = TestClient(app).get("/api/v1/missing")

    assert response.status_code == 404
