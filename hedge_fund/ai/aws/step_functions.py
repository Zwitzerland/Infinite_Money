"""Step Functions helpers."""
from __future__ import annotations

import json
from typing import Any, Mapping

from .clients import get_client


def start_execution(
    state_machine_arn: str,
    payload: Mapping[str, Any],
    region: str | None = None,
    profile: str | None = None,
) -> str:
    """Start a Step Functions execution."""
    client = get_client("stepfunctions", region=region, profile=profile)
    response = client.start_execution(
        stateMachineArn=state_machine_arn,
        input=json.dumps(payload),
    )
    return str(response["executionArn"])


def describe_execution(
    execution_arn: str,
    region: str | None = None,
    profile: str | None = None,
) -> Mapping[str, Any]:
    """Describe a Step Functions execution."""
    client = get_client("stepfunctions", region=region, profile=profile)
    return client.describe_execution(executionArn=execution_arn)
