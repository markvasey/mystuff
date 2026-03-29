#!/bin/bash
# Script to stop the SpringBootConsole app

PORT=8081
PID=$(lsof -ti:$PORT)

if [ -z "$PID" ]; then
  echo "No application running on port $PORT"
else
  echo "Stopping application running on port $PORT (PID: $PID)..."
  kill $PID
  echo "Application stopped successfully."
fi
