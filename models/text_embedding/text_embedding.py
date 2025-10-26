from __future__ import annotations

import time
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

import json
import httpx

from dify_plugin.entities.model import EmbeddingInputType
from dify_plugin.entities.model.text_embedding import EmbeddingUsage, TextEmbeddingResult
from dify_plugin.interfaces.model.text_embedding_model import TextEmbeddingModel

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _headers(api_key: str) -> dict:
    return {
        "api-key": api_key,
        "Content-Type": "application/json",
    }


def _client_with_read_timeout(seconds: float) -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(connect=10.0, read=seconds))


def _post_with_retry(
    client: httpx.Client,
    url: str,
    headers: dict,
    payload: dict,
    max_retries: int = 3,
) -> httpx.Response:
    backoff = 0.5
    for attempt in range(max_retries + 1):
        try:
            response = client.post(url, headers=headers, json=payload)
            if response.status_code in _RETRYABLE_STATUS:
                raise httpx.HTTPStatusError(
                    "retryable status",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            return response
        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout):
            if attempt >= max_retries:
                raise
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("Exceeded retry attempts")


def _to_decimal(value: Any, default: str = "0.0") -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)


def _usage_from_dict(source: Any) -> EmbeddingUsage:
    if not isinstance(source, dict):
        return EmbeddingUsage(
            tokens=0,
            total_tokens=0,
            unit_price=Decimal("0.0"),
            price_unit=Decimal("0.0"),
            total_price=Decimal("0.0"),
            currency="USD",
            latency=0.0,
        )
    tokens = int(source.get("tokens") or 0)
    total_tokens = int(source.get("total_tokens") or tokens)
    return EmbeddingUsage(
        tokens=tokens,
        total_tokens=total_tokens,
        unit_price=_to_decimal(source.get("unit_price")),
        price_unit=_to_decimal(source.get("price_unit")),
        total_price=_to_decimal(source.get("total_price")),
        currency=str(source.get("currency") or "USD"),
        latency=float(source.get("latency") or 0.0),
    )

def _parse_gateway_error_text(text: str) -> dict:
    """
    いろいろなゲートウェイ/SDKが返す error ボディをできるだけ正規化して抽出する。
    対応パターン:
      - {"error": {"code": "...", "message": "..."}}
      - {"error": {"type": "...", "message": "...", "code": "..."}}
      - {"message": "..."} または {"detail": "..."}
      - 二重JSON（例: {"message": "{\"error_type\":\"TypeError\",\"message\":\"...\"}"}）
    """
    def _coerce_obj(s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    obj = _coerce_obj(text) if isinstance(text, str) else None
    if not obj:
        return {"code": "unknown_error", "message": text or ""}

    # azure/openai系: {"error": {...}}
    if isinstance(obj, dict) and "error" in obj and isinstance(obj["error"], dict):
        err = obj["error"]
        code = err.get("code") or err.get("type") or "unknown_error"
        msg = err.get("message") or err.get("msg") or err.get("detail") or ""
        return {"code": str(code), "message": str(msg)}

    # {"message": "..."} or {"detail": "..."} にネストJSONが入っているケース
    for k in ("message", "detail"):
        if k in obj:
            inner = obj[k]
            if isinstance(inner, str):
                nested = _coerce_obj(inner)
                if isinstance(nested, dict):
                    # {"error_type": "...", "message": "..."} など
                    code = nested.get("code") or nested.get("error_type") or "unknown_error"
                    msg = nested.get("message") or nested.get("detail") or inner
                    return {"code": str(code), "message": str(msg)}
                return {"code": "error", "message": inner}
            elif isinstance(inner, dict):
                code = inner.get("code") or inner.get("type") or "error"
                msg = inner.get("message") or inner.get("detail") or ""
                return {"code": str(code), "message": str(msg)}

    # それ以外は丸ごと
    return {"code": "unknown_error", "message": text if isinstance(text, str) else json.dumps(obj, ensure_ascii=False)}

class AzureCompatibleEmbedding(TextEmbeddingModel):
    def _invoke(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        user: Optional[str] = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        base_url = credentials["base_url"].rstrip("/")
        endpoint = f"{base_url}/openai/deployments/{model}/embeddings?api-version=fake"
        headers = _headers(credentials["api_key"])
        timeout = float(credentials.get("timeout_sync", 60))
        payload: dict[str, Any] = {"input": texts}
        if user:
            payload["user"] = user
        if input_type:
            payload["input_type"] = input_type.value if hasattr(input_type, "value") else str(input_type)

        with _client_with_read_timeout(timeout) as client:
            response = _post_with_retry(client, endpoint, headers, payload)
            data = response.json()

        vectors: list[list[float]] = []
        if isinstance(data, dict):
            if isinstance(data.get("data"), Sequence):
                for item in data["data"]:
                    if isinstance(item, dict) and isinstance(item.get("embedding"), list):
                        vectors.append(item["embedding"])
            elif isinstance(data.get("embeddings"), Sequence):
                for embedding in data["embeddings"]:
                    if isinstance(embedding, list):
                        vectors.append(embedding)

        usage = _usage_from_dict(data.get("usage"))
        return TextEmbeddingResult(model=model, embeddings=vectors, usage=usage)

    def get_num_tokens(self, model: str, credentials: dict, texts: list[str]) -> list[int]:
        # Basic heuristic: use character length as token approximation when tokenizer info is unavailable.
        return [len(text.encode("utf-8")) for text in texts]
    
    def _invoke_error_mapping(self, exc: Exception) -> Exception:
        import httpx
        if isinstance(exc, (httpx.ConnectError, httpx.ProxyError)):
            return ConnectionError(f"network_unreachable: {exc}")
        if isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
            return TimeoutError(f"timeout: {exc}")
        if isinstance(exc, httpx.HTTPStatusError):
            r = exc.response
            sc = r.status_code
            info = _parse_gateway_error_text(r.text)
            code = (info.get("code") or "http_error").lower()
            msg = info.get("message") or r.text
            if code in ("429", "rate_limit_exceeded", "rate_limit", "too_many_requests", "insufficient_quota"):
                code = "rate_limited"
            if code in ("deploymentnotfound", "model_not_found", "not_found"):
                code = "model_not_found"
            if code in ("invalid_request_error", "validation_error", "bad_request"):
                code = "invalid_request"
            norm = f"{sc} {code}: {msg}".strip()
            if sc == 400 or code == "invalid_request":
                return ValueError(norm)
            if sc in (401, 403):
                return PermissionError(norm)
            if sc == 404 or code == "model_not_found":
                return FileNotFoundError(norm)
            if sc == 409:
                return RuntimeError(f"conflict: {norm}")
            if sc == 422:
                return ValueError(f"validation_error: {norm}")
            if sc == 429 or code == "rate_limited":
                return RuntimeError(f"rate_limited: {norm}")
            if 500 <= sc < 600:
                return RuntimeError(f"upstream_error: {norm}")
            return RuntimeError(norm)
        return exc

    def invoke_error_mapping(self, exc: Exception) -> Exception:  # pragma: no cover
        return self._invoke_error_mapping(exc)
   
