from __future__ import annotations

import json
import math
import time
from collections.abc import Generator, Iterable, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

from dify_plugin.entities.model.llm import (
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
    LLMUsage,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageContent,
    PromptMessageTool,
)
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeConnectionError,
    InvokeError,
)
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel

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


def _pseudo_sse_chunks(text: str, parts: int) -> Iterable[str]:
    text = text or ""
    parts = max(1, int(parts))
    if parts == 1:
        yield text
        return
    if not text:
        yield ""
        return
    stride = max(1, math.ceil(len(text) / parts))
    for start in range(0, len(text), stride):
        yield text[start : start + stride]


def _iter_sse_payload(stream: httpx.Response) -> Iterable[str]:
    buffer: list[str] = []
    for raw_line in stream.iter_lines():
        if raw_line is None:
            continue
        line = raw_line.strip()
        if not line:
            if buffer:
                yield "\n".join(buffer)
                buffer.clear()
            continue
        if line.startswith("data:"):
            data = line[5:].strip()
            if data == "[DONE]":
                break
            if data:
                buffer.append(data)
    if buffer:
        yield "\n".join(buffer)


def _serialize_message_content(content: Any) -> Any:
    if isinstance(content, list):
        serialized = []
        for item in content:
            if isinstance(item, PromptMessageContent):
                serialized.append(item.model_dump(mode="json"))
            elif hasattr(item, "model_dump"):
                serialized.append(item.model_dump(mode="json"))  # type: ignore[attr-defined]
            elif isinstance(item, dict):
                serialized.append(item)
        return serialized
    if hasattr(content, "model_dump"):
        return content.model_dump(mode="json")  # type: ignore[attr-defined]
    return content


def _serialize_messages(messages: Sequence[PromptMessage | dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, PromptMessage):
            payload: dict[str, Any] = {"role": message.role.value}
            if message.name:
                payload["name"] = message.name
            payload["content"] = _serialize_message_content(message.content)
            result.append(payload)
        elif isinstance(message, dict):
            result.append(message)
        else:
            raise TypeError(f"Unsupported message type: {type(message)!r}")
    return result


def _serialize_tools(tools: Optional[Sequence[PromptMessageTool | dict[str, Any]]]) -> Optional[list[dict[str, Any]]]:
    if not tools:
        return None
    serialized: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, PromptMessageTool):
            serialized.append(tool.model_dump(mode="json"))
        elif isinstance(tool, dict):
            serialized.append(tool)
        else:
            raise TypeError(f"Unsupported tool type: {type(tool)!r}")
    return serialized


def _to_decimal(value: Any, default: str = "0.0") -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)


def _usage_from_dict(source: Any) -> LLMUsage:
    if not isinstance(source, dict):
        return LLMUsage.empty_usage()
    prompt_tokens = int(source.get("prompt_tokens") or 0)
    completion_tokens = int(source.get("completion_tokens") or 0)
    total_tokens = int(source.get("total_tokens") or (prompt_tokens + completion_tokens))
    total_price = _to_decimal(source.get("total_price"), default="0.0")
    return LLMUsage(
        prompt_tokens=prompt_tokens,
        prompt_unit_price=_to_decimal(source.get("prompt_unit_price")),
        prompt_price_unit=_to_decimal(source.get("prompt_price_unit")),
        prompt_price=_to_decimal(source.get("prompt_price")),
        completion_tokens=completion_tokens,
        completion_unit_price=_to_decimal(source.get("completion_unit_price")),
        completion_price_unit=_to_decimal(source.get("completion_price_unit")),
        completion_price=_to_decimal(source.get("completion_price")),
        total_tokens=total_tokens,
        total_price=total_price,
        currency=str(source.get("currency") or "USD"),
        latency=float(source.get("latency") or 0.0),
    )


class AzureCompatibleLLM(LargeLanguageModel):
    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        return {
            InvokeConnectionError: [
                httpx.ConnectError,
                httpx.RemoteProtocolError,
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
            ],
            InvokeError: [httpx.HTTPStatusError, httpx.HTTPError, RuntimeError, ValueError],
        }

    def validate_credentials(self, model: str, credentials: dict) -> None:
        base_url = self._require_str(credentials, "base_url")
        self._validate_url(base_url)
        self._require_str(credentials, "api_key")
        self._ensure_float(credentials.get("timeout_sync"), "timeout_sync")
        self._ensure_float(credentials.get("timeout_async"), "timeout_async")
        self._ensure_positive_int(credentials.get("pseudo_sse_chunks"), "pseudo_sse_chunks", allow_empty=True)

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Generator[LLMResultChunk, None, None] | LLMResult:
        if stream:
            return self._invoke_stream(
                model=model,
                credentials=credentials,
                prompt_messages=prompt_messages,
                model_parameters=model_parameters,
                tools=tools,
                stop=stop,
                user=user,
            )
        return self._invoke_non_stream(
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            model_parameters=model_parameters,
            tools=tools,
            stop=stop,
            user=user,
        )

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        return sum(self._count_prompt_tokens(message) for message in prompt_messages)

    def _invoke_non_stream(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]],
        stop: Optional[list[str]],
        user: Optional[str],
    ) -> LLMResult:
        base_url = credentials["base_url"].rstrip("/")
        endpoint = f"{base_url}/openai/deployments/{model}/responses?api-version=fake"
        headers = _headers(credentials["api_key"])
        timeout = float(credentials.get("timeout_sync", 60))
        payload = self._build_payload(
            model=model,
            prompt_messages=prompt_messages,
            tools=tools,
            model_parameters=model_parameters,
            stream=False,
            stop=stop,
            user=user,
        )

        with _client_with_read_timeout(timeout) as client:
            response = _post_with_retry(client, endpoint, headers, payload)
            data = response.json()

        text = self._extract_text_from_responses(data)
        if stop and text:
            text = self.enforce_stop_tokens(text, stop)
        message = AssistantPromptMessage(content=text)
        usage = _usage_from_dict(data.get("usage"))
        system_fingerprint = data.get("system_fingerprint")

        return LLMResult(
            model=model,
            prompt_messages=prompt_messages,
            message=message,
            usage=usage,
            system_fingerprint=system_fingerprint,
        )

    def _invoke_stream(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]],
        stop: Optional[list[str]],
        user: Optional[str],
    ) -> Generator[LLMResultChunk, None, None]:
        base_url = credentials["base_url"].rstrip("/")
        endpoint = f"{base_url}/openai/deployments/{model}/responses?api-version=fake"
        headers = _headers(credentials["api_key"])
        timeout_sync = float(credentials.get("timeout_sync", 60))
        timeout_stream = float(credentials.get("timeout_async", 300))
        pseudo_chunks = max(int(credentials.get("pseudo_sse_chunks", 2) or 2), 1)
        is_openai = self._is_openai(model)

        payload = self._build_payload(
            model=model,
            prompt_messages=prompt_messages,
            tools=tools,
            model_parameters=model_parameters,
            stream=is_openai,
            stop=stop,
            user=user,
        )

        if is_openai:
            return self._stream_openai_responses(
                model=model,
                endpoint=endpoint,
                headers=headers,
                payload=payload,
                prompt_messages=prompt_messages,
                timeout=timeout_stream,
                stop=stop,
            )

        return self._stream_pseudo_responses(
            model=model,
            endpoint=endpoint,
            headers=headers,
            payload=payload,
            prompt_messages=prompt_messages,
            timeout=timeout_sync,
            stop=stop,
            chunks=pseudo_chunks,
        )

    def _stream_openai_responses(
        self,
        *,
        model: str,
        endpoint: str,
        headers: dict,
        payload: dict,
        prompt_messages: list[PromptMessage],
        timeout: float,
        stop: Optional[list[str]],
    ) -> Generator[LLMResultChunk, None, None]:
        def generator() -> Generator[LLMResultChunk, None, None]:
            accumulated = ""
            emitted = ""
            index = 0
            usage = LLMUsage.empty_usage()

            with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=timeout)) as client:
                with client.stream("POST", endpoint, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    for event_payload in _iter_sse_payload(response):
                        delta_text, usage_candidate = self._parse_sse_event(event_payload)
                        if usage_candidate is not None:
                            usage = usage_candidate
                        if not delta_text:
                            continue
                        accumulated += delta_text
                        processed = self.enforce_stop_tokens(accumulated, stop) if stop else accumulated
                        new_output = processed[len(emitted) :]
                        if not new_output:
                            if processed != accumulated:
                                break
                            continue
                        finish_reason = None
                        if processed != accumulated:
                            finish_reason = "stop"
                        emitted = processed
                        yield LLMResultChunk(
                            model=model,
                            prompt_messages=prompt_messages,
                            delta=LLMResultChunkDelta(
                                index=index,
                                message=AssistantPromptMessage(content=new_output),
                            ),
                        )
                        index += 1
                        if finish_reason:
                            yield LLMResultChunk(
                                model=model,
                                prompt_messages=prompt_messages,
                                delta=LLMResultChunkDelta(
                                    index=index,
                                    message=AssistantPromptMessage(content=""),
                                    finish_reason=finish_reason,
                                    usage=usage,
                                ),
                            )
                            return

            yield LLMResultChunk(
                model=model,
                prompt_messages=prompt_messages,
                delta=LLMResultChunkDelta(
                    index=index,
                    message=AssistantPromptMessage(content=""),
                    finish_reason="stop",
                    usage=usage,
                ),
            )

        return generator()

    def _stream_pseudo_responses(
        self,
        *,
        model: str,
        endpoint: str,
        headers: dict,
        payload: dict,
        prompt_messages: list[PromptMessage],
        timeout: float,
        stop: Optional[list[str]],
        chunks: int,
    ) -> Generator[LLMResultChunk, None, None]:
        def generator() -> Generator[LLMResultChunk, None, None]:
            with _client_with_read_timeout(timeout) as client:
                response = _post_with_retry(client, endpoint, headers, payload)
                data = response.json()

            full_text = self._extract_text_from_responses(data)
            if stop and full_text:
                full_text = self.enforce_stop_tokens(full_text, stop)
            usage = _usage_from_dict(data.get("usage"))
            index = 0
            emitted_any = False

            for segment in _pseudo_sse_chunks(full_text, chunks):
                emitted_any = True
                yield LLMResultChunk(
                    model=model,
                    prompt_messages=prompt_messages,
                    delta=LLMResultChunkDelta(
                        index=index,
                        message=AssistantPromptMessage(content=segment),
                    ),
                )
                index += 1

            if not emitted_any:
                yield LLMResultChunk(
                    model=model,
                    prompt_messages=prompt_messages,
                    delta=LLMResultChunkDelta(
                        index=index,
                        message=AssistantPromptMessage(content=""),
                    ),
                )

            yield LLMResultChunk(
                model=model,
                prompt_messages=prompt_messages,
                delta=LLMResultChunkDelta(
                    index=index + 1,
                    message=AssistantPromptMessage(content=""),
                    finish_reason="stop",
                    usage=usage,
                ),
            )

        return generator()

    def _parse_sse_event(self, payload: str) -> tuple[str, LLMUsage | None]:
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return payload, None

        if isinstance(event, dict):
            event_type = str(event.get("type") or "")
            if "error" in event_type:
                detail = event.get("error") or event
                raise RuntimeError(f"SSE stream error: {detail}")
            usage_candidate = event.get("usage") or event.get("response", {}).get("usage")
            text_candidate = self._extract_text_from_responses(event, fallback_to_json=False)
            usage = _usage_from_dict(usage_candidate) if usage_candidate else None
            return text_candidate, usage
        return "", None

    def _is_openai(self, model: str) -> bool:
        return model.startswith("gpt-") or model.startswith("o3")

    def _build_payload(
        self,
        *,
        model: str,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]],
        model_parameters: dict,
        stream: bool,
        stop: Optional[list[str]],
        user: Optional[str],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "input": {"messages": _serialize_messages(prompt_messages)},
            "stream": stream,
        }
        if user:
            payload["input"]["user"] = user
        serialized_tools = _serialize_tools(tools)
        if serialized_tools:
            payload["tools"] = serialized_tools

        temperature = model_parameters.get("temperature")
        top_p = model_parameters.get("top_p")
        max_tokens = model_parameters.get("max_output_tokens")
        json_response = bool(model_parameters.get("json_response"))
        json_schema = model_parameters.get("json_schema")
        thinking = bool(model_parameters.get("thinking"))
        enable_vision = bool(model_parameters.get("enable_vision"))

        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop
        if enable_vision:
            payload["input"]["modalities"] = ["text", "vision"]
        if json_response:
            payload["response_format"] = {"type": "json_object"}
            if json_schema:
                try:
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": json.loads(json_schema),
                    }
                except (TypeError, json.JSONDecodeError):
                    pass
        if thinking and self._is_openai(model):
            payload["reasoning"] = {"effort": "medium"}
        return payload

    def _extract_text_from_responses(self, data: Any, *, fallback_to_json: bool = True) -> str:
        if data is None:
            return ""
        if isinstance(data, str):
            return data
        if isinstance(data, list):
            combined = "".join(
                self._extract_text_from_responses(item, fallback_to_json=fallback_to_json) for item in data
            )
            return combined
        if not isinstance(data, dict):
            return ""

        for key in ("text", "output_text", "content"):
            value = data.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return "".join(
                    self._extract_text_from_responses(item, fallback_to_json=fallback_to_json) for item in value
                )

        if "message" in data and isinstance(data["message"], dict):
            nested = self._extract_text_from_responses(data["message"], fallback_to_json=fallback_to_json)
            if nested:
                return nested

        choices = data.get("choices")
        if isinstance(choices, list):
            fragments = []
            for choice in choices:
                if isinstance(choice, dict):
                    fragments.append(
                        self._extract_text_from_responses(choice.get("delta"), fallback_to_json=fallback_to_json)
                    )
                    fragments.append(
                        self._extract_text_from_responses(choice.get("message"), fallback_to_json=fallback_to_json)
                    )
            text = "".join(fragments)
            if text:
                return text

        outputs = data.get("output") or data.get("outputs")
        if isinstance(outputs, list):
            concatenated = "".join(
                self._extract_text_from_responses(item, fallback_to_json=fallback_to_json) for item in outputs
            )
            if concatenated:
                return concatenated

        if fallback_to_json:
            return json.dumps(data, ensure_ascii=False)
        return ""

    def _count_prompt_tokens(self, message: PromptMessage) -> int:
        content = message.content
        if isinstance(content, str):
            return len(content.encode("utf-8"))
        if isinstance(content, list):
            total = 0
            for item in content:
                if isinstance(item, PromptMessageContent):
                    data = getattr(item, "data", "") or getattr(item, "text", "")
                    if isinstance(data, str):
                        total += len(data.encode("utf-8"))
                    elif isinstance(data, bytes):
                        total += len(data)
                elif isinstance(item, dict):
                    total += len(json.dumps(item, ensure_ascii=False).encode("utf-8"))
                elif hasattr(item, "model_dump"):
                    total += len(
                        json.dumps(item.model_dump(mode="json"), ensure_ascii=False).encode("utf-8")
                    )  # type: ignore[attr-defined]
                else:
                    total += len(str(item).encode("utf-8"))
            return total
        if content is None:
            return 0
        if hasattr(content, "model_dump"):
            return len(json.dumps(content.model_dump(mode="json"), ensure_ascii=False).encode("utf-8"))  # type: ignore[attr-defined]
        return len(str(content).encode("utf-8"))

    @staticmethod
    def _require_str(credentials: dict, key: str) -> str:
        value = credentials.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise CredentialsValidateFailedError(f"Missing credential: {key}")

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise CredentialsValidateFailedError("base_url must be a valid HTTP(S) URL.")

    @staticmethod
    def _ensure_float(value: Any, field: str) -> None:
        if value in (None, ""):
            return
        try:
            float(value)
        except (TypeError, ValueError):
            raise CredentialsValidateFailedError(f"Invalid numeric value for {field}.") from None

    @staticmethod
    def _ensure_positive_int(value: Any, field: str, *, allow_empty: bool = False) -> None:
        if value in (None, ""):
            if allow_empty:
                return
            raise CredentialsValidateFailedError(f"{field} is required.")
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            raise CredentialsValidateFailedError(f"{field} must be an integer.") from None
        if candidate < 1:
            raise CredentialsValidateFailedError(f"{field} must be >= 1.")
