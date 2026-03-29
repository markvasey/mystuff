#!/bin/bash
# Script to verify start_app.sh and stop_app.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
START_SCRIPT="$SCRIPT_DIR/start_app.sh"
STOP_SCRIPT="$SCRIPT_DIR/stop_app.sh"
PID_FILE="$SCRIPT_DIR/app.pid"
PORT=8081

echo "--- Starting Script Verification ---"

# 1. Ensure we start from a clean state
echo "[1/4] Ensuring clean state..."
$STOP_SCRIPT > /dev/null 2>&1
sleep 2

# 2. Test Start Script
echo "[2/4] Testing start_app.sh..."
$START_SCRIPT > /dev/null 2>&1

if [ ! -f "$PID_FILE" ]; then
    echo "  FAILED: app.pid was not created."
    exit 1
fi

PID=$(cat "$PID_FILE")
if ! ps -p $PID > /dev/null; then
    echo "  FAILED: Process $PID is not running."
    exit 1
fi
echo "  SUCCESS: Process $PID is running."

# 3. Wait for port to open (up to 30 seconds)
echo "[3/4] Waiting for port $PORT to open (this may take a moment)..."
MAX_ATTEMPTS=30
ATTEMPT=1
while ! lsof -ti:$PORT > /dev/null; do
    if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
        echo "  FAILED: Port $PORT did not open after $MAX_ATTEMPTS seconds."
        exit 1
    fi
    sleep 1
    ((ATTEMPT++))
done
echo "  SUCCESS: Port $PORT is listening."

# 4. Test Stop Script
echo "[4/4] Testing stop_app.sh..."
$STOP_SCRIPT > /dev/null 2>&1
sleep 2

if ps -p $PID > /dev/null; then
    echo "  FAILED: Process $PID is still running after stop_app.sh."
    exit 1
fi

if lsof -ti:$PORT > /dev/null; then
    echo "  FAILED: Port $PORT is still listening after stop_app.sh."
    exit 1
fi
echo "  SUCCESS: Application stopped and port $PORT is free."

echo "--- All Tests Passed! ---"
