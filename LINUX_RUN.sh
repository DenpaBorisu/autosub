#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

# Check for uv
if ! command -v uv &>/dev/null; then
    echo "uv is not installed. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "uv installed."
fi

uv sync
uv run python autosub_gui.py "$@"
