# MCP Servers

Model Context Protocol servers expose controlled tooling for the agent control
plane. Each server should provide a small, audited surface for external system
access (QuantConnect, Secrets Manager, storage, quantum jobs).

## Structure

- Implement each MCP server as a focused module under `mcp_servers/`.
- Keep request/response shapes typed and deterministic.
- Route all secret access through the secrets MCP server (no plaintext configs).
