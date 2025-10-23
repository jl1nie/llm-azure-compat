from __future__ import annotations

import time
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

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
