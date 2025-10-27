# Azure-Compatible Gateway Agent Notes

This plugin wraps any Azure-style gateway that exposes OpenAI-compatible `responses`/`chat.completions` and `embeddings` routes. Use it when your Dify agent needs to talk to a single endpoint that fans out to multiple models by deployment ID.

## Capabilities at a Glance
- **LLM streaming** – native SSE for OpenAI family models (`gpt-*`, `o3`); pseudo-SSE chunking for Claude and Gemini so that downstream consumers always see incremental updates.
- **Structured output** – forwards `json_schema`, `tool_choice`, and tool definitions exactly how Dify sends them, after sanity-normalising the schema and parameters. For chat/completions targets, JSON-only mode cannot be enforced upstream.
- **Reasoning toggle** – sets `reasoning.effort=medium` when a request carries `thinking` metadata.
- **Embeddings** – routes through the same credential set, mapping the deployment ID to the embedding model name.

## Credential Form Reference
| Field | Required | Default | Notes |
|-------|----------|---------|-------|
| `base_url` | ✓ | — | Must include the scheme (`https://gateway.example.com`). |
| `api_key` | ✓ | — | Sent as the `api-key` header; no `Bearer` prefix. |
| `api_version` | ✗ | `2025-10-21` (UI) / `2024-02-15-preview` (runtime fallback) | Append as `?api-version=...` when `use_deployid=true`. |
| `use_deployid` | ✗ | `true` | Flip to `false` if your gateway exposes `/chat/completions` without the `/deployments/{model}` segment. |
| `timeout_sync` | ✗ | `60` | Read timeout for non-streaming calls, in seconds. |
| `timeout_async` | ✗ | `300` | Read timeout for streaming calls. |
| `pseudo_sse_chunks` | ✗ | `2` | How many slices to emit when simulating streaming for non-OpenAI models; must be ≥ 1. |

### URL Patterns
- `use_deployid=true` → `POST {base_url}/openai/deployments/{deployment_id}/chat/completions?api-version={api_version}`
- `use_deployid=false` → `POST {base_url}/chat/completions`
- Embeddings follow the same toggle but hit `/embeddings` instead of `/chat/completions`.

## Supported Models
| Type | Identifier | Notes |
|------|------------|-------|
| LLM | `gpt-5`, `gpt-4.1`, `o3` | Native OpenAI streaming. |
| LLM | `claude-4.1` | Pseudo-streamed; set higher chunk count for longer replies. |
| LLM | `gemini-2.5-pro`, `gemini-2.5-flash` | Pseudo-streamed; structured hints are forwarded but not enforced. |
| LLM | `qwen3-30b-a3b`, `gpt-oss-20b` | Non-OpenAI, pseudo-streamed. |
| Embedding | `text-embedding-3-large`, `text-embedding-3-small`, `text-embedding-ada-002` | Same credential scope as LLM models. |

> **Tip:** Deployment IDs must match the model name unless your gateway rewrites them internally.

## Prompt & Tooling Guidance
- The serializer preserves complex multimodal content blocks and tool calls; you can safely attach vision or function-calling instructions without extra wrapping.
- The provider now validates tool definitions locally; malformed function specs raise immediately instead of producing Azure 400 responses.
- For agents that rely heavily on tool calling, prefer OpenAI family models where possible; pseudo-streamed providers will still return tool calls correctly, but the response arrives once the upstream request finishes.
- When you need structured replies, provide a `json_schema` hint; the gateway passes it through unchanged, though enforcement depends on the upstream model.
- When invoking “reasoning” style prompts (o3, gpt-5 “thinking”), include the standard Dify reasoning metadata so the gateway sets `reasoning.effort=medium`.

## Error Surface
- Network issues (`ConnectError`, TLS, timeouts) surface as `ConnectionError` or `TimeoutError`.
- HTTP errors are normalised: missing deployments raise `FileNotFoundError`, rate limits become `RuntimeError` with `rate_limited`, validation issues translate to `ValueError`.
- The gateway attempts to decode nested JSON error messages (common with reverse proxies) so agent logs remain actionable.

## Operational Checklist
1. Verify the gateway answers `/openai/deployments/<model>/chat/completions` before wiring into Dify.
2. Create one provider credential in Dify and share it across both LLM and embedding nodes.
3. Test streaming with `gpt-4.1` or `gpt-5`; then confirm pseudo-stream behaviour using `claude-4.1` if applicable.
4. Adjust `pseudo_sse_chunks` when downstream consumers expect more granular updates (e.g., real-time UIs).
5. Monitor logs for repeated `rate_limited` errors—consider increasing backoff or upstream quotas if they appear.
