from __future__ import annotations

import json
import math
import time
from collections.abc import Generator, Iterable
from typing import Any, Dict, Optional, Union

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
    PromptMessageTool,
)
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel


def _headers(api_key: str) -> dict:
    # Azure/OpenAI compatible gateway: use api-key header instead of Bearer
    return {"api-key": api_key, "Content-Type": "application/json"}


def _client_read_timeout(seconds: float) -> httpx.Client:
    # Use system CA certificates; do not override verify.
    return httpx.Client(timeout=httpx.Timeout(read=seconds, connect=10.0, write=10.0, pool=10.0))


def _post_with_retry(
    client: httpx.Client,
    url: str,
    headers: dict,
    json_body: dict,
    max_retries: int = 3,
) -> httpx.Response:
    backoff = 0.5
    for attempt in range(max_retries + 1):
        try:
            response = client.post(url, headers=headers, json=json_body)
            if response.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError("retryable", request=response.request, response=response)
            response.raise_for_status()
            return response
        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout):
            if attempt >= max_retries:
                raise
            time.sleep(backoff)
            backoff *= 2


def _stream_sse(url: str, headers: dict, payload: dict, stream_timeout: float) -> Iterable[str]:
    """Yield 'data: ...' payloads from an SSE response as UTF-8 strings."""
    timeout = httpx.Timeout(read=stream_timeout, connect=10.0, write=10.0, pool=10.0)
    with httpx.Client(timeout=timeout) as sse_client:
        with sse_client.stream("POST", url, headers=headers, json=payload) as response:
            response.raise_for_status()
            buffer = b""
            for raw in response.iter_raw():
                if not raw:
                    continue
                buffer += raw
                while b"\n\n" in buffer:
                    chunk, buffer = buffer.split(b"\n\n", 1)
                    for line in chunk.split(b"\n"):
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            return
                        yield data.decode("utf-8", errors="ignore")

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

def _as_dict(obj: Any, *, default: dict | None = None) -> dict:
    """Pydantic / dataclass / mapping / JSON文字列 / その他 -> dict に正規化"""
    if obj is None:
        return {} if default is None else default
    try:
        if hasattr(obj, "model_dump"):
            d = obj.model_dump(mode="json", exclude_none=True)
            return d if isinstance(d, dict) else {}
        if hasattr(obj, "dict"):
            d = obj.dict(exclude_none=True)
            return d if isinstance(d, dict) else {}
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            try:
                d = json.loads(obj)
                return d if isinstance(d, dict) else {}
            except Exception:
                return {} if default is None else default
        # callable や未知のオブジェクトは弾く
        return {} if default is None else default
    except Exception:
        return {} if default is None else default

def _as_number_map(obj: Any) -> dict[str, float]:
    """OpenAI 互換の logit_bias など: dict[str, float] 以外は空に"""
    d = _as_dict(obj)
    out: dict[str, float] = {}
    for k, v in d.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out

class AzureCompatibleLLM(LargeLanguageModel):
    """
    Azure-like Responses gateway.
    {base_url}/openai/deployments/{deployment_id}/chat/completions?api-version={api_version}
    OpenAI models (gpt-*, o3) stream natively; others simulate streaming.
    """

    def _invoke(
        self,
        model: str,
        credentials: Dict[str, str],
        prompt_messages: list[PromptMessage],
        model_parameters: Optional[dict] = None,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator[LLMResultChunk, None, None]]:
                
        mp = model_parameters or {}
        serialized_messages = self._serialize_messages(prompt_messages)
        serialized_tools = self._serialize_tools(tools)
        is_openai = self._is_openai(model)
        payload = self._build_chat_payload(
            messages=serialized_messages,
            tools=serialized_tools,
            mp=mp,
            stream=bool(stream and is_openai),
            stop=stop,
        )

        base_url = credentials["base_url"].rstrip("/")
        use_deployid = bool(credentials.get("use_deployid","true") == "true")
        if use_deployid:
            api_version = credentials.get("api_version", "2024-02-15-preview")
            url = f"{base_url}/openai/deployments/{model}/chat/completions?api-version={api_version}"
        else:
            url = f"{base_url}/chat/completions?"
        headers = _headers(credentials["api_key"])
        timeout_sync = self._get_timeout(credentials.get("timeout_sync"), default=300.0)
        timeout_stream = self._get_timeout(credentials.get("timeout_async"), default=1800.0)
        pseudo_chunks = max(int(credentials.get("pseudo_sse_chunks") or 2), 1)

        if stream:
            if is_openai:
                return self._iter_native_stream(
                    url=url,
                    headers=headers,
                    payload=payload,
                    model=model,
                    prompt_messages=prompt_messages,
                    stream_timeout=timeout_stream,
                )
            return self._iter_pseudo_stream(
                url=url,
                headers=headers,
                payload=payload,
                model=model,
                prompt_messages=prompt_messages,
                timeout=timeout_sync,
                chunk_count=pseudo_chunks,
            )

        try:
            with _client_read_timeout(timeout_sync) as client:
                response = _post_with_retry(client, url, headers, payload)
                data = response.json()
        except Exception as e:
            raise self._invoke_error_mapping(e)

        text = self._extract_text(data)
        usage, has_usage = self._usage_from_dict(data.get("usage"))
        message = AssistantPromptMessage(content=text)
        result_model = data.get("model", model)
        fingerprint = data.get("system_fingerprint")
        return LLMResult(
            model=result_model,
            prompt_messages=prompt_messages,
            message=message,
            usage=usage if has_usage else LLMUsage.empty_usage(),
            system_fingerprint=fingerprint,
        )

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        del model, credentials, tools  # Token counting is heuristic only
        total_chars = 0
        for message in prompt_messages:
            for fragment in self._message_text_fragments(message):
                total_chars += len(fragment)
        if total_chars <= 0:
            return max(1, len(prompt_messages))
        return max(1, (total_chars + 3) // 4)  # Rough 4 chars ≈ 1 token heuristic
    
    def validate_credentials(self, credentials: dict) -> None:
        base = (credentials.get("base_url") or "").strip()
        if not base:
            raise ValueError("Missing credential: base_url")
        if not credentials.get("api_key"):
            raise ValueError("Missing credential: api_key")
        # 形式簡易チェック
        if base.startswith("http://") or base.startswith("https://"):
            pass
        else:
            raise ValueError("base_url must start with http:// or https://")
        for k in ("timeout_sync", "timeout_async"):
            v = credentials.get(k)
            if v not in (None, ""):
                float(v)
        pcs = credentials.get("pseudo_sse_chunks")
        if pcs not in (None, ""):
            n = int(pcs)
            if n < 1:
                raise ValueError("pseudo_sse_chunks must be >= 1")


    # ---------- stream helpers ----------
    def _iter_native_stream(
        self,
        *,
        url: str,
        headers: dict,
        payload: dict,
        model: str,
        prompt_messages: list[PromptMessage],
        stream_timeout: float,
    ) -> Generator[LLMResultChunk, None, None]:
        def iterator() -> Generator[LLMResultChunk, None, None]:
            try:
                for event in _stream_sse(url, headers, payload, stream_timeout):
                    try:
                        obj = json.loads(event)
                    except json.JSONDecodeError:
                        yield self._build_chunk(
                            model=model,
                            prompt_messages=prompt_messages,
                            system_fingerprint=None,
                            index=0,
                            content=event,
                            finish_reason=None,
                            usage=None,
                        )
                        continue

                    event_model = obj.get("model", model)
                    fingerprint = obj.get("system_fingerprint")
                    usage_dict = obj.get("usage")
                    usage, has_usage = self._usage_from_dict(usage_dict)

                    for choice in obj.get("choices", []):
                        index = choice.get("index", 0)
                        delta = choice.get("delta") or {}
                        finish_reason = choice.get("finish_reason")

                        if delta.get("content"):
                            yield self._build_chunk(
                                model=event_model,
                                prompt_messages=prompt_messages,
                                system_fingerprint=fingerprint,
                                index=index,
                                content=delta["content"],
                                finish_reason=None,
                                usage=None,
                            )

                        if finish_reason:
                            yield self._build_chunk(
                                model=event_model,
                                prompt_messages=prompt_messages,
                                system_fingerprint=fingerprint,
                                index=index,
                                content="",
                                finish_reason=finish_reason,
                                usage=usage if has_usage else None,
                            )
            except Exception as e:
                raise self._invoke_error_mapping(e)
        return iterator()

    def _iter_pseudo_stream(
        self,
        *,
        url: str,
        headers: dict,
        payload: dict,
        model: str,
        prompt_messages: list[PromptMessage],
        timeout: float,
        chunk_count: int,
    ) -> Generator[LLMResultChunk, None, None]:
        def iterator() -> Generator[LLMResultChunk, None, None]:
            try:
                with _client_read_timeout(timeout) as client:
                    response = _post_with_retry(client, url, headers, payload)
                    data = response.json()
            except Exception as e:
                raise self._invoke_error_mapping(e)

            text = self._extract_text(data)
            usage, has_usage = self._usage_from_dict(data.get("usage"))
            fingerprint = data.get("system_fingerprint")
            response_model = data.get("model", model)

            if not text:
                yield self._build_chunk(
                    model=response_model,
                    prompt_messages=prompt_messages,
                    system_fingerprint=fingerprint,
                    index=0,
                    content="",
                    finish_reason="stop",
                    usage=usage if has_usage else None,
                )
                return

            segment_size = max(1, math.ceil(len(text) / chunk_count))
            total_segments = math.ceil(len(text) / segment_size)
            for idx, start in enumerate(range(0, len(text), segment_size)):
                segment = text[start : start + segment_size]
                is_last = idx == total_segments - 1
                yield self._build_chunk(
                    model=response_model,
                    prompt_messages=prompt_messages,
                    system_fingerprint=fingerprint,
                    index=idx,
                    content=segment,
                    finish_reason="stop" if is_last else None,
                    usage=usage if is_last and has_usage else None,
                )
        return iterator()

    # ---------- serialization helpers ----------
    def _serialize_messages(self, prompt_messages: list[PromptMessage]) -> list[dict]:
        serialized: list[dict] = []
        for message in prompt_messages:
            if hasattr(message, "model_dump"):
                raw = message.model_dump(mode="json", exclude_none=True)
            elif hasattr(message, "dict"):
                raw = message.dict(exclude_none=True)
            elif isinstance(message, dict):
                raw = dict(message)
            else:
                raw = {
                    "role": getattr(message, "role", None),
                    "content": getattr(message, "content", None),
                }

            if isinstance(raw.get("content"), list):
                raw["content"] = [self._normalize_content_item(item) for item in raw["content"]]
            if isinstance(raw.get("tool_calls"), list):
                raw["tool_calls"] = [self._normalize_tool_call(call) for call in raw["tool_calls"]]
            serialized.append(raw)
        return serialized

    def _normalize_content_item(self, item: Any) -> Any:
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json", exclude_none=True)
        if hasattr(item, "dict"):
            return item.dict(exclude_none=True)
        if isinstance(item, dict):
            return item
        return item

    def _normalize_tool_call(self, tool_call: Any) -> dict:
        # どんな型でも dict へ潰す（callableは除去）
        d = _as_dict(tool_call)
        # 既に OpenAI 形式なら minimum validation
        t = d.get("type")
        if t == "function" and isinstance(d.get("function"), (dict,)):
            fn = d["function"]
            # parameters を dict 化（callable / 文字列 JSON も潰す）
            fn["parameters"] = _as_dict(fn.get("parameters"))
            # name/description が callable など不正な場合は文字列化
            name = fn.get("name")
            if not isinstance(name, str):
                fn["name"] = str(name) if name is not None else "function"
            desc = fn.get("description")
            if desc is not None and not isinstance(desc, str):
                fn["description"] = str(desc)
            d["function"] = fn
            return d

        # 旧式や独自型 → OpenAI 形式に合成
        name = d.get("name") or getattr(tool_call, "name", None) or "function"
        description = d.get("description") or getattr(tool_call, "description", None)
        parameters = d.get("parameters")
        if parameters is None:
            parameters = getattr(tool_call, "parameters", None)
        parameters = _as_dict(parameters)  # ここで callable を排除

        return {
            "type": "function",
            "function": {
                "name": name if isinstance(name, str) else "function",
                "description": description if isinstance(description, str) else None,
                "parameters": parameters,
            },
        }

    def _serialize_tools(self, tools: Optional[list[PromptMessageTool]]) -> Optional[list[dict]]:
        if not tools:
            return None
        out: list[dict] = []
        for tool in tools:
            if tool is None:
                continue
            # どんな入力でも _normalize_tool_call で強制整形
            try:
                out.append(self._normalize_tool_call(tool))
            except Exception as e:
                # skip invalid tool
                continue
        return out or None

    # ---------- payload helpers ----------
    def _build_chat_payload(
        self,
        *,
        messages: list[dict],
        tools: Optional[list[dict]],
        mp: dict,
        stream: bool,
        stop: Optional[list[str]],
    ) -> dict:
        max_tokens = mp.get("max_tokens", mp.get("max_output_tokens"))
        body: Dict[str, Any] = {
            "messages": messages,
            "stream": stream,
        }

        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if mp.get("temperature") is not None:
            body["temperature"] = mp["temperature"]
        if mp.get("top_p") is not None:
            body["top_p"] = mp["top_p"]
        if stop:
            body["stop"] = stop
        if mp.get("presence_penalty") is not None:
            body["presence_penalty"] = mp["presence_penalty"]
        if mp.get("frequency_penalty") is not None:
            body["frequency_penalty"] = mp["frequency_penalty"]
        if mp.get("logit_bias") is not None:
            body["logit_bias"] = mp["logit_bias"]
        if mp.get("n") is not None:
            body["n"] = mp["n"]
        if mp.get("seed") is not None:
            body["seed"] = mp["seed"]
        if mp.get("tool_choice") is not None:
            body["tool_choice"] = mp["tool_choice"]
        if mp.get("json_response"):
            body["response_format"] = {"type": "json_object"}
            schema = mp.get("json_schema")
            if schema:
                try:
                    if isinstance(schema, str):
                        schema = json.loads(schema)
                    body["response_format"] = {"type": "json_schema", "json_schema": schema}
                except Exception:
                    pass
        if mp.get("thinking"):
            body["reasoning"] = {"effort": "medium"}

        if tools:
            body["tools"] = tools
            functions = []
            for item in tools:
                if isinstance(item, dict) and item.get("type") == "function" and isinstance(item.get("function"), dict):
                    functions.append(item["function"])
            if functions:
                body["functions"] = functions
                body.setdefault("function_call", "auto")

        return body

    # ---------- utility helpers ----------
    @staticmethod
    def _is_openai(model: str) -> bool:
        return model.startswith("gpt-") or model.startswith("o3")

    @staticmethod
    def _get_timeout(value: Any, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _message_text_fragments(self, message: PromptMessage) -> Iterable[str]:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        yield text
                else:
                    text = getattr(item, "text", None)
                    if isinstance(text, str):
                        yield text

    @staticmethod
    def _usage_from_dict(usage: Any) -> tuple[Optional[LLMUsage], bool]:
        """
        SDK バージョン差異に耐える LLMUsage 構築ヘルパ。
        - LLMUsage.from_metadata が無い環境でも動く
        - LLMUsage.empty_usage も不要にする
        - 生成できない場合は (None, False) を返す
        """
        # 正規化
        meta: Dict[str, Any] = {}
        if isinstance(usage, dict):
            # よく使うキーをそのまま拾う＋将来の拡張に備えて残りは metadata として保持
            for k in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "prompt_unit_price",
                "completion_unit_price",
                "total_price",
                "currency",
                "prompt_price_unit",
                "completion_price_unit",
                "prompt_price",
                "completion_price",
                "latency",
            ):
                if usage.get(k) is not None:
                    meta[k] = usage[k]
            # その他のキーも落とさず持っておく
            for k, v in usage.items():
                if k not in meta:
                    meta[k] = v
        elif usage is None:
            meta = {}
        else:
            # dict 以外は一旦文字列化して格納
            meta = {"raw": str(usage)}

        # いくつかのコンストラクタ候補で試す
        candidates = [
            # 代表的: 個別トークン数を与える型
            dict(
                prompt_tokens=meta.get("prompt_tokens"),
                completion_tokens=meta.get("completion_tokens"),
                total_tokens=meta.get("total_tokens"),
                metadata=meta,
            ),
            # metadata だけ与える型
            dict(metadata=meta),
            # そのまま渡せる型（将来の互換）
            meta,
            # 引数なし
            dict(),
        ]

        for params in candidates:
            # None は渡さない（TypeError を避ける）
            clean = {k: v for k, v in params.items() if v is not None}
            try:
                obj = LLMUsage(**clean)  # type: ignore
                return obj, True
            except TypeError:
                continue
            except Exception:
                continue

        # どうしても作れない場合
        return None, False


    def _build_chunk(
        self,
        *,
        model: str,
        prompt_messages: list[PromptMessage],
        system_fingerprint: Optional[str],
        index: int,
        content: str,
        finish_reason: Optional[str],
        usage: Optional[LLMUsage],
    ) -> LLMResultChunk:
        return LLMResultChunk(
            model=model,
            prompt_messages=prompt_messages,
            system_fingerprint=system_fingerprint,
            delta=LLMResultChunkDelta(
                index=index,
                message=AssistantPromptMessage(content=content),
                finish_reason=finish_reason,
                usage=usage,
            ),
        )

    def _extract_text(self, data: Dict[str, Any]) -> str:
        try:
            if isinstance(data.get("output"), list):
                segments: list[str] = []
                for item in data["output"]:
                    for content in item.get("content", []):
                        text = content.get("text") if isinstance(content, dict) else None
                        if isinstance(text, str):
                            segments.append(text)
                if segments:
                    return "".join(segments)

            if isinstance(data.get("choices"), list):
                message_text = "".join(
                    choice.get("message", {}).get("content", "") for choice in data["choices"]
                )
                if message_text:
                    return message_text
                delta_text = "".join(choice.get("delta", {}).get("content", "") for choice in data["choices"])
                if delta_text:
                    return delta_text
        except Exception:
            pass

        if isinstance(data.get("content"), str):
            return data["content"]
        if isinstance(data.get("output_text"), str):
            return data["output_text"]
        return json.dumps(data, ensure_ascii=False)
    
    def _invoke_error_mapping(self, exc: Exception) -> Exception:
        """
        Normalize httpx/network/HTTP errors into readable exceptions.
        This name (with leading underscore) matches the ABC in some SDK versions.
        """
        import httpx

        # Network-level
        if isinstance(exc, (httpx.ConnectError, httpx.ProxyError)):
            return ConnectionError(f"network_unreachable: {exc}")
        if isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
            return TimeoutError(f"timeout: {exc}")

        # HTTP status
        if isinstance(exc, httpx.HTTPStatusError):
            r = exc.response
            sc = r.status_code
            info = _parse_gateway_error_text(r.text)
            code = (info.get("code") or "http_error").lower()
            msg = info.get("message") or r.text

            # normalize common codes
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

        # default passthrough
        return exc

    # ---- compatibility: some SDKs expect invoke_error_mapping (no underscore) ----
    def invoke_error_mapping(self, exc: Exception) -> Exception:  # pragma: no cover
        return self._invoke_error_mapping(exc)
