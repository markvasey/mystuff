#!/bin/bash
# Template for start_app.sh

# Set Java Environment
export JAVA_HOME=/home/markvasey/.sdkman/candidates/java/26.0.1-tem
export PATH=$JAVA_HOME/bin:$PATH

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="$APP_DIR/scripts/app.log"
PID_FILE="$APP_DIR/scripts/app.pid"

echo "Starting application..."
cd "$APP_DIR" || exit
nohup ./mvnw spring-boot:run > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Application started in background (PID: $!). Logging to $LOG_FILE"
