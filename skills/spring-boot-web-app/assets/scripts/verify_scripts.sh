#!/bin/bash
# Template to verify start_app.sh and stop_app.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
START_SCRIPT="$SCRIPT_DIR/start_app.sh"
STOP_SCRIPT="$SCRIPT_DIR/stop_app.sh"
PID_FILE="$SCRIPT_DIR/app.pid"
PORT=8081

echo "--- Starting Script Verification ---"

# 1. Ensure clean state
$STOP_SCRIPT > /dev/null 2>&1
sleep 2

# 2. Test Start
$START_SCRIPT > /dev/null 2>&1
if [ ! -f "$PID_FILE" ]; then echo "FAILED: app.pid not created"; exit 1; fi
PID=$(cat "$PID_FILE")
if ! ps -p $PID > /dev/null; then echo "FAILED: Process $PID not running"; exit 1; fi

# 3. Wait for port
MAX_ATTEMPTS=30; ATTEMPT=1
while ! lsof -ti:$PORT > /dev/null; do
    if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then echo "FAILED: Port $PORT did not open"; exit 1; fi
    sleep 1; ((ATTEMPT++))
done

# 4. Test Stop
$STOP_SCRIPT > /dev/null 2>&1
sleep 2
if ps -p $PID > /dev/null; then echo "FAILED: Process still running"; exit 1; fi

echo "--- All Tests Passed! ---"
