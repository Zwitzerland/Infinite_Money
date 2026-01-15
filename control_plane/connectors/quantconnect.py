"""QuantConnect Cloud connector.

Notes
-----
This module defines request/response shapes for QuantConnect API operations and
provides a minimal HTTP client implementation. API tokens must be injected via
secrets management; do not hardcode credentials.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import requests
from requests import Response

from hedge_fund.utils.settings import PlatformSettings


@dataclass(frozen=True)
class CompilationRequest:
    """Compilation request payload."""

    project_id: int
    name: str


@dataclass(frozen=True)
class BacktestRequest:
    """Backtest creation request payload."""

    project_id: int
    name: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class LiveRequest:
    """Live algorithm deployment request payload."""

    project_id: int
    brokerage: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class FileCreateRequest:
    """File creation/update request payload."""

    project_id: int
    name: str
    content: str


class QuantConnectClient:
    """Typed connector for QuantConnect Cloud endpoints."""

    def __init__(self, base_url: str, api_token: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_token = api_token

    @classmethod
    def from_settings(cls, settings: PlatformSettings) -> "QuantConnectClient":
        """Create a client from platform settings.

        Raises
        ------
        ValueError
            If the API token is missing.
        """
        token = settings.quantconnect_api_token
        if token is None:
            raise ValueError(
                "QuantConnect API token missing; inject via secrets manager."
            )
        return cls(settings.quantconnect_base_url, token.get_secret_value())

    def _request(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        url = f"{self._base_url}{path}"
        response = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {self._api_token}"},
            timeout=30,
        )
        self._raise_for_status(response)
        return response.json()

    @staticmethod
    def _raise_for_status(response: Response) -> None:
        if response.ok:
            return
        raise RuntimeError(
            "QuantConnect API request failed with "
            f"{response.status_code}: {response.text}"
        )

    def create_compile_job(self, payload: CompilationRequest) -> Mapping[str, Any]:
        """Create a compilation job."""
        return self._request("/compile/create", payload.__dict__)

    def create_backtest(self, payload: BacktestRequest) -> Mapping[str, Any]:
        """Create a backtest job.

        Raises
        ------
        RuntimeError
            If the request fails.
        """
        return self._request("/backtests/create", payload.__dict__)

    def create_live_algorithm(self, payload: LiveRequest) -> Mapping[str, Any]:
        """Create a live algorithm deployment."""
        return self._request("/live/create", payload.__dict__)

    def create_file(self, payload: FileCreateRequest) -> Mapping[str, Any]:
        """Create or update a file in the project."""
        return self._request("/files/create", payload.__dict__)
