"""A dropped model body must not reappear in OpenAI-compatible responses."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server import _create_app, _make_adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("path,payload", [
    ("/v1/chat/completions", {"model": "hermes-agent", "messages": [{"role": "user", "content": "input"}]}),
    ("/v1/responses", {"model": "hermes-agent", "input": "input"}),
])
async def test_dropped_candidate_is_not_reconstructed_by_api_surface(path, payload):
    sentinel = "FORK1B_DROP_SECRET"
    result = {
        "final_response": None,
        "completed": False,
        "failed": True,
        "output_disposition": "dropped",
        "failure_reason": "final_output_dropped",
        "error": "final_output_dropped",
        "messages": [{"role": "user", "content": "input"}],
        "api_calls": 1,
    }
    assert sentinel not in str(result)
    adapter = _make_adapter()
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as client:
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as run_agent:
            run_agent.return_value = (result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
            response = await client.post(path, json=payload)
            body = await response.text()
    assert sentinel not in body
    assert sentinel not in json.dumps(dict(response.headers))
    assert body
