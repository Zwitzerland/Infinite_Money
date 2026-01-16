# MCP Server Map

This document outlines the intended Model Context Protocol server surface.

## Required servers

1. `mcp_quantconnect_cloud`
   - compile/create
   - backtests/create
   - live/create
   - files/create
2. `mcp_storage`
   - list artifacts
   - write artifacts
3. `mcp_secrets`
   - retrieve secret handles (no plaintext)
4. `mcp_braket`
   - submit hybrid jobs
   - poll job status

## Guardrails

- Secrets retrieval should always be audited and logged.
- No MCP server may place trades directly; trading must flow through QuantConnect
  live deployments only after promotion gates pass.
