"""Verify local browser assets and restrictive response headers."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/", "text/html"),
        ("/static/app.js", "text/javascript"),
        ("/static/styles.css", "text/css"),
    ],
)
def test_browser_assets_are_local_and_not_cached(path: str, content_type: str) -> None:
    client = TestClient(app, base_url="http://localhost")
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(content_type)
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    csp = response.headers["content-security-policy"]
    assert "default-src 'none'" in csp
    assert "script-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert client.get(path, headers={"host": "evil.example"}).status_code == 403


def test_unlisted_static_files_are_not_served() -> None:
    client = TestClient(app, base_url="http://localhost")
    assert client.get("/static/unknown.js").status_code == 404
