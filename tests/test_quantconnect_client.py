from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from control_plane.connectors.quantconnect import CompilationRequest, QuantConnectClient
from hedge_fund.utils.settings import PlatformSettings


def test_client_from_settings_requires_token() -> None:
    settings = PlatformSettings(quantconnect_api_token=None)
    with pytest.raises(ValueError, match="QuantConnect API token missing"):
        QuantConnectClient.from_settings(settings)


def test_create_compile_job_posts_to_expected_endpoint() -> None:
    client = QuantConnectClient("https://example.test/api", "token-123")
    response = Mock()
    response.ok = True
    response.json.return_value = {"success": True}
    with patch("control_plane.connectors.quantconnect.requests.post", return_value=response) as post:
        payload = CompilationRequest(project_id=1, name="demo")
        result = client.create_compile_job(payload)

    post.assert_called_once()
    called_args, called_kwargs = post.call_args
    assert called_args[0] == "https://example.test/api/compile/create"
    assert called_kwargs["headers"]["Authorization"] == "Bearer token-123"
    assert result["success"] is True
