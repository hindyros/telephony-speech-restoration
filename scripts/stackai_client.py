"""
stackai_client.py — thin HTTP wrapper around the Stack AI inference bridge.

Usage
-----
    from scripts.stackai_client import StackAIClient

    client = StackAIClient(token="<your-token>")
    text   = client.query("gpt_5_4", "Correct this transcript: ...")

In Google Colab, load the token from Colab Secrets:
    from google.colab import userdata
    client = StackAIClient(token=userdata.get('STACKAI_TOKEN'))

Environment variables (optional — override hardcoded endpoint IDs)
-------------------------------------------------------------------
    STACKAI_TOKEN          Bearer token (required if not passed to __init__)
    STACKAI_RUN_ID         Override the default run ID
    STACKAI_GPT54PRO_EP    Override endpoint for gpt_5_4_pro
    STACKAI_GPT54_EP       Override endpoint for gpt_5_4
    STACKAI_GPT4O_EP       Override endpoint for gpt_4o
    STACKAI_SONNET46_EP    Override endpoint for sonnet_4_6
    STACKAI_OPUS46_EP      Override endpoint for opus_4_6
    STACKAI_GEMINI35_EP    Override endpoint for gemini_3_5
    STACKAI_HAIKU45_EP     Override endpoint for haiku_4_5
"""
from __future__ import annotations

import os
import time

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STACKAI_BASE = "https://api.stackai.com/inference/v0/run/{run_id}/{endpoint_id}"

_DEFAULT_RUN_ID = "475c0540-6417-4995-ae46-62b1f9b8fe4a"

MODEL_REGISTRY: dict[str, str] = {
    "gpt_5_4_pro": os.getenv("STACKAI_GPT54PRO_EP", "69e7e0aef65bb6b52b53a555"),
    "gpt_5_4":     os.getenv("STACKAI_GPT54_EP",    "69e7e168bea566f9d2a4109b"),
    "gpt_4o":      os.getenv("STACKAI_GPT4O_EP",    "69e7e1ab55e57fa84fc09ec2"),
    "sonnet_4_6":  os.getenv("STACKAI_SONNET46_EP", "69e7e2339505c1211767cbd4"),
    "opus_4_6":    os.getenv("STACKAI_OPUS46_EP",   "69e7e326e31c3a23d8f8b33c"),
    "gemini_3_5":  os.getenv("STACKAI_GEMINI35_EP", "69e7e35dbea566f9d2a4109c"),
    "haiku_4_5":   os.getenv("STACKAI_HAIKU45_EP",  "69e7e470a265535ccb488e95"),
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class StackAIClient:
    """HTTP client for the Stack AI inference bridge."""

    def __init__(
        self,
        token: str | None = None,
        run_id: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        timeout: float = 120.0,
    ) -> None:
        self._token = token or os.getenv("STACKAI_TOKEN", "")
        if not self._token:
            raise ValueError(
                "STACKAI_TOKEN is not set. "
                "Pass it to StackAIClient(token=...) or set the env var."
            )
        self._run_id = run_id or os.getenv("STACKAI_RUN_ID", _DEFAULT_RUN_ID)
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, model_name: str, prompt: str, user_id: str = "") -> str:
        """
        Send a single-turn text prompt to the named model.

        Parameters
        ----------
        model_name : key from MODEL_REGISTRY (e.g. ``"gpt_5_4"``)
        prompt     : user message text
        user_id    : optional session identifier passed to Stack AI

        Returns
        -------
        Response text as a plain string.

        Raises
        ------
        ValueError     — unknown model name or unparseable response
        RuntimeError   — all retries exhausted
        """
        endpoint_id = self._endpoint_id(model_name)
        url = STACKAI_BASE.format(run_id=self._run_id, endpoint_id=endpoint_id)
        payload = {"user_id": user_id, "in-0": prompt}

        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = requests.post(
                    url, headers=self._headers, json=payload, timeout=self._timeout
                )
                # 4xx = caller error — don't retry
                if 400 <= resp.status_code < 500:
                    resp.raise_for_status()
                # 5xx = server error — retry
                if resp.status_code >= 500:
                    raise requests.HTTPError(
                        f"HTTP {resp.status_code}", response=resp
                    )
                return self._extract_text(resp.json())
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    print(
                        f"  [stackai] attempt {attempt} failed ({exc}), "
                        f"retrying in {self._retry_delay}s..."
                    )
                    time.sleep(self._retry_delay)

        raise RuntimeError(
            f"Stack AI query failed after {self._max_retries} attempts: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _endpoint_id(self, model_name: str) -> str:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {sorted(MODEL_REGISTRY)}"
            )
        return MODEL_REGISTRY[model_name]

    @staticmethod
    def _extract_text(data: dict) -> str:
        """
        Try multiple JSON shapes to extract the response string.
        Stack AI may return different structures depending on the flow version.
        """
        # Most common: {"outputs": {"out-0": "..."}}
        if isinstance(data.get("outputs"), dict):
            out = data["outputs"].get("out-0")
            if isinstance(out, str):
                return out

        # Flat: {"out-0": "..."}
        if isinstance(data.get("out-0"), str):
            return data["out-0"]

        # Generic fallbacks
        for key in ("response", "text", "content", "message", "result"):
            if isinstance(data.get(key), str):
                return data[key]

        raise ValueError(
            f"Cannot extract text from Stack AI response. "
            f"Top-level keys: {list(data.keys())}"
        )
