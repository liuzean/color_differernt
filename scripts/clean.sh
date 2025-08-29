#!/bin/bash
# Simple wrapper script for cleaning cache and temp files

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🧹 Cleaning cache and temporary files..."
echo "Project root: $PROJECT_ROOT"
echo

# Run the cleanup script
python "$SCRIPT_DIR/cleanup.py" --root "$PROJECT_ROOT" "$@"

echo
echo "✨ Cleanup complete!" 