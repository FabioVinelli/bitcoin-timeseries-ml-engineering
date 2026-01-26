# Glassnode MCP Server

> The Glassnode MCP Server is currently in beta, and its tools are subject to change. Although this MCP server produces accurate data, it is possible that it could be misinterpreted by the LLM you are using.

## Overview

*Crypto on-chain metrics and market intelligence for AI agents.*

A Model Context Protocol (MCP) server providing access to [Glassnode's](https://glassnode.com/) API. The server enables AI agents to access on-chain metrics, market data, and analytics.

For detailed information about the API and available data:

* [Glassnode API Documentation](https://docs.glassnode.com/basic-api/endpoints)
* [Glassnode Metric Catalog](https://docs.glassnode.com/data/metric-catalog)
* [Supported Assets](https://docs.glassnode.com/data/supported-assets)

**Features**

* Asset and metrics discovery
* Metric metadata retrieval
* Single and bulk data retrieval

**Tools**

The server provides seven tools for accessing Glassnode data:

1. `get_assets_list`: Get a list of all cryptocurrencies and tokens supported by Glassnode
2. `get_metrics_list`: Get a catalog of all available metrics and their paths, optionally filtered by category
3. `get_asset_metrics`: Get metrics available for a specific asset
4. `get_metric_metadata`: Get detailed information about a specific metric's structure and parameters
5. `fetch_metric`: Fetch data (max 30d history) for a specific metric with customizable parameters
6. `fetch_bulk_metrics`: Fetch data (max 30d history) for multiple assets simultaneously in a single request

## Quick Setup

### Claude

To add the Glassnode MCP server using Claude Connectors:

1. Open Claude desktop or web app
2. Go to `Settings > Connectors > Add custom connector` (<https://claude.ai/settings/connectors>)
3. Enter the URL: `https://mcp.glassnode.com`
4. Click `Add`
5. Claude will automatically configure the connection

For more advanced configuration, read below.

### Claude Desktop Configuration

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "glassnode": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://mcp.glassnode.com"
      ]
    }
  }
}
```

Or using your API key, to remove 30d time range limits:

```json
{
  "mcpServers": {
    "glassnode": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://mcp.glassnode.com",
        "--header",
        "X-Api-Key:${GLASSNODE_API_KEY}"
      ],
      "env": {
        "GLASSNODE_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

### Claude CLI Quick Add

```bash
# Public access
claude mcp add-json glassnode '{
  "command": "npx",
  "args": ["mcp-remote", "https://mcp.glassnode.com"]
}'

# With API key
claude mcp add-json glassnode '{
  "command": "npx",
  "args": [
    "mcp-remote",
    "https://mcp.glassnode.com",
    "--header",
    "X-Api-Key:${GLASSNODE_API_KEY}"
  ],
  "env": {
    "GLASSNODE_API_KEY": "your-api-key"
  }
}'
```

Restart your Claude client after updating configs.

### Other MCP clients

Add the following configuration to your `mcp_config.json`:

```jsx
{
  "mcpServers": {
    "glassnode": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://mcp.glassnode.com"
      ]
    }
  }
}
```

## Example Prompts

Get an overview of the MCP tools.

```jsx
What can Glassnode mcp do?
```

Simple prompts:

```jsx
- What market metrics Glassnode has for Ethereum?
- What assets does Glassnode support?
- Bitcoin price, use 10 minute resolution
```

More advanced:

```jsx
- SOPR and MVRV for Bitcoin and Ethereum, values for the past month
- Show me Bitcoin whale activity
- Compare the market caps of Bitcoin, Ethereum, and Solana over the last month
```

***
