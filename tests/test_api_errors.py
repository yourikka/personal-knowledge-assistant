from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from app.api_errors import api_error, register_error_handlers


class ExampleRequest(BaseModel):
    name: str = Field(min_length=1)


def build_client() -> TestClient:
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/missing")
    def missing():
        raise api_error(404, "document_not_found", "文档不存在。", retryable=False)

    @app.post("/validate")
    def validate(_: ExampleRequest):
        return {"status": "ok"}

    return TestClient(app)


def test_api_error_uses_structured_payload():
    response = build_client().get("/missing")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "document_not_found",
            "message": "文档不存在。",
            "retryable": False,
        }
    }


def test_validation_error_uses_structured_payload():
    response = build_client().post("/validate", json={"name": ""})
    payload = response.json()

    assert response.status_code == 422
    assert payload["error"]["code"] == "request_validation_failed"
    assert payload["error"]["retryable"] is False
    assert "name" in payload["error"]["message"]
