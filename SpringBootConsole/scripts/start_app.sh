#!/bin/bash
# Script to start the SpringBootConsole app

# Set Java Environment
export JAVA_HOME=/home/markvasey/.jdks/openjdk-23.0.1
export PATH=$JAVA_HOME/bin:$PATH

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="$APP_DIR/scripts/app.log"
PID_FILE="$APP_DIR/scripts/app.pid"

echo "Starting SpringBootConsole on port 8081..."
cd "$APP_DIR" || exit
nohup ./mvnw spring-boot:run > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Application started in background (PID: $!). Logging to $LOG_FILE"
