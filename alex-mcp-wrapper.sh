#!/bin/bash
# Wrapper script for alex-mcp that activates the virtual environment

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Export environment variables for MCP server
export OPENALEX_ENABLE_SCIHUB=true

# Run the MCP server
exec python -m alex_mcp.server "$@"
