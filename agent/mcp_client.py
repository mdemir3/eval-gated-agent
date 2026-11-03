"""MCP client the Researcher uses to call tools.

`MCPToolClient.stdio()` launches `tools/mcp_server/server.py` as a subprocess
and speaks MCP over stdin/stdout. Every call returns a `ToolResult`; transport
errors and tool errors are reported as data, never raised, so one bad call
can't crash an agent run.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client

REPO_ROOT = Path(__file__).resolve().parents[1]
# Server stderr (model-loading chatter, tool logs) goes here instead of the terminal.
SERVER_LOG = Path(os.getenv("MCP_SERVER_LOG") or Path(tempfile.gettempdir()) / "eval-gated-agent-mcp.log")

# Least privilege: the tool server gets the SDK's safe default env plus these,
# and never the LLM API keys.
_PASSTHROUGH_ENV = (
    "CHROMA_DIR",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "SENTENCE_TRANSFORMERS_HOME",
    "TRANSFORMERS_OFFLINE",
    "XDG_CACHE_HOME",
)
_QUIET_ENV = {
    # The embedding model is downloaded by `evals.build_index`; the server only reads the cache.
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "TQDM_DISABLE": "1",
    "TORCH_CPP_LOG_LEVEL": "ERROR",
    "PYTHONWARNINGS": "ignore",
}


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    text: str = ""
    error: str | None = None


def server_params() -> StdioServerParameters:
    env = get_default_environment()
    env.update(_QUIET_ENV)
    env.update({k: os.environ[k] for k in _PASSTHROUGH_ENV if k in os.environ})
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "tools.mcp_server.server"],
        cwd=str(REPO_ROOT),
        env=env,
    )


class MCPToolClient:
    """Async context manager around an MCP `Client` session."""

    def __init__(self, server: Any = None) -> None:
        # `server` is an in-process MCPServer (tests); None launches the stdio subprocess.
        self._server = server
        self._client: Client | None = None
        self._errlog: Any = None

    @classmethod
    def stdio(cls) -> MCPToolClient:
        return cls()

    async def __aenter__(self) -> MCPToolClient:
        target = self._server
        if target is None:
            self._errlog = open(SERVER_LOG, "a", encoding="utf-8")
            target = stdio_client(server_params(), errlog=self._errlog)
        self._client = Client(target, read_timeout_seconds=120)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        try:
            if self._client is not None:
                await self._client.__aexit__(*exc)
        finally:
            self._client = None
            if self._errlog is not None:
                self._errlog.close()
                self._errlog = None

    async def list_tools(self) -> list[ToolSpec]:
        assert self._client is not None, "use `async with`"
        result = await self._client.list_tools()
        return [ToolSpec(t.name, t.description or "", t.input_schema) for t in result.tools]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        assert self._client is not None, "use `async with`"
        try:
            result = await self._client.call_tool(name, arguments)
        except Exception as exc:  # transport failure, timeout, protocol error
            return ToolResult(ok=False, error=f"MCP call failed: {type(exc).__name__}: {exc}")
        text = "\n".join(getattr(block, "text", "") for block in result.content).strip()
        if result.is_error:
            return ToolResult(ok=False, text=text, error=text or "tool returned an error")
        data = result.structured_content
        if data is None:
            try:
                data = json.loads(text)
            except ValueError:
                data = text
        return ToolResult(ok=True, data=data, text=text)
