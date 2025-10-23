# Azure-Compatible Gateway Provider

An Azure-styled gateway for Dify that maps deployment IDs directly to model names and speaks the OpenAI **Responses** and **Embeddings** APIs with a fixed `api-version=fake`. The provider exposes Large Language Models (LLM) and text embedding models behind a single credential set.

## Features
- Native SSE streaming for OpenAI models (`gpt-*`, `o3`) with automatic parsing of Responses events.
- Pseudo-SSE streaming for non-OpenAI chat models (Claude, Gemini) with configurable chunk counts.
- Transparent passthrough for JSON mode (`json_response`, `json_schema`) and reasoning requests (`thinking → reasoning.effort=medium`).
- Shared credential form: `base_url`, `api_key`, per-call timeouts, and pseudo-SSE chunk count.
- Vision toggles: `enable_vision` simply forwards the vision modality flag while messages remain untouched.

## Supported Models
- **LLM:** `gpt-5`, `gpt-4.1`, `o3`, `claude-4.1`, `gemini-2.5-pro`, `gemini-2.5-flash`
- **Text Embedding:** `text-embedding-3-large`, `text-embedding-3-small`, `text-embedding-ada-002`

All endpoints follow the Azure OpenAI pattern:

```
POST {base_url}/openai/deployments/{deployment_id}/responses?api-version=fake
POST {base_url}/openai/deployments/{deployment_id}/embeddings?api-version=fake
```

The deployment ID is expected to match the model name.

## Configuration
Within Dify, create a provider instance with:

1. **Base URL** – the root of your Azure-compatible gateway (no trailing slash).
2. **API Key** – sent as `api-key: <key>` without the `Bearer` prefix.
3. Optional **timeouts** — synchronous (default 60s) and streaming (default 300s) read timeouts.
4. Optional **pseudo-SSE chunk count** — how many slices to emit for non-OpenAI responses (default 2).

## Local Development
```
pyenv install -s 3.13.8
pyenv local 3.13.8
uv venv --python 3.13.8
uv pip install -e .
python -c "import models.llm.llm, models.text_embedding.text_embedding; print('ok')"
```

## Notes
- TLS verification relies on the container image (`/etc/ssl`); no custom CA handling is bundled.
- The provider is validation-only: no outbound credential checks are executed during setup.
