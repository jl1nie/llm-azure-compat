from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from dify_plugin import ModelProvider
from dify_plugin.errors.model import CredentialsValidateFailedError


class AzureCompatibleProvider(ModelProvider):
    """Provider validator for Azure-compatible gateway credentials."""

    def validate_provider_credentials(self, credentials: Mapping[str, Any]) -> None:
        base_url = self._require_str(credentials, "base_url")
        self._validate_url(base_url)
        self._require_str(credentials, "api_key")

        self._ensure_float(credentials.get("timeout_sync"), "timeout_sync")
        self._ensure_float(credentials.get("timeout_async"), "timeout_async")
        self._ensure_positive_int(credentials.get("pseudo_sse_chunks"), "pseudo_sse_chunks", allow_empty=True)

    @staticmethod
    def _require_str(credentials: Mapping[str, Any], key: str) -> str:
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
