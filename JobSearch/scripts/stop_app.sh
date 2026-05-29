#!/bin/bash
# stop_app.sh

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$APP_DIR/scripts/app.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "Stopping application (PID: $PID)..."
    kill "$PID"
    rm "$PID_FILE"
    echo "Application stopped."
else
    echo "No PID file found. Is the application running?"
fi
