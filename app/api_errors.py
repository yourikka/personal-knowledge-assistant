from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


def error_payload(code: str, message: str, retryable: bool = False) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        }
    }


def api_error(status_code: int, code: str, message: str, retryable: bool = False) -> HTTPException:
    return HTTPException(status_code=status_code, detail=error_payload(code, message, retryable))


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def structured_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
            content = detail
        else:
            content = error_payload("http_error", str(detail), retryable=exc.status_code >= 500)
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(RequestValidationError)
    async def structured_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first_error = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first_error.get("loc", []) if str(part))
        message = str(first_error.get("msg") or "请求参数校验失败。")
        if location:
            message = f"{location}: {message}"
        return JSONResponse(
            status_code=422,
            content=error_payload("request_validation_failed", message, retryable=False),
        )
