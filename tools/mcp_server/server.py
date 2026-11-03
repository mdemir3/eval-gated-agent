"""MCP tool server (stdio) exposing `rag_search` and `mock_api_get`.

Run standalone with `python -m tools.mcp_server.server`; the agent launches it
as a subprocess and talks to it over the MCP protocol. Tool failures are
returned as MCP error results (`is_error=True`), never as a crashed server.
"""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from tools.mcp_server.mock_data import ENDPOINTS, MockAPI, MockAPIError, resolve_path
from tools.mcp_server.rag import Retriever

_ENDPOINT_DOC = "\n".join(f"  {path}: {desc}" for path, desc in ENDPOINTS.items())


def create_server(retriever: Retriever | None = None, api: MockAPI | None = None) -> MCPServer:
    """Build a server instance. Each instance has its own mock-API state."""
    retriever = retriever or Retriever()
    api = api or MockAPI()
    server = MCPServer(
        name="eval-gated-agent-tools",
        instructions="Tools for a demo QA agent: corpus search and a deterministic fake REST API.",
        log_level="WARNING",
    )

    @server.tool(
        description=(
            "Search a small general-knowledge corpus (capitals, planets, inventions, geography). "
            "Returns the top_k most similar chunks with their doc ids and cosine similarity scores."
        )
    )
    def rag_search(
        query: Annotated[str, Field(min_length=1, description="Natural-language search query")],
        top_k: Annotated[int, Field(ge=1, le=10, description="Number of chunks to return (1-10)")] = 3,
    ) -> dict[str, Any]:
        try:
            hits = retriever.search(query, top_k)
        except FileNotFoundError as exc:
            raise ToolError(str(exc)) from None
        return {"query": query, "results": hits}

    @server.tool(
        description=(
            "GET a record from a deterministic demo REST API (fake data). Pass a concrete path such as "
            "'/users/3', or a template such as '/users/{id}' with params {'id': 3}. Endpoints:\n"
            + _ENDPOINT_DOC
        )
    )
    def mock_api_get(
        endpoint: Annotated[str, Field(min_length=1, description="API path, e.g. /weather/tokyo")],
        params: Annotated[
            dict[str, str | int] | None, Field(description="Values for {placeholders} in the endpoint")
        ] = None,
    ) -> dict[str, Any]:
        try:
            path = resolve_path(endpoint, params)
            data = api.get(path)
        except MockAPIError as exc:
            raise ToolError(f"GET {endpoint} failed: {exc}") from None
        return {"endpoint": path, "status": 200, "data": data}

    return server


def main() -> None:
    create_server().run("stdio")


if __name__ == "__main__":
    main()
