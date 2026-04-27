#!/bin/bash
# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

export JAVA_HOME=/home/markvasey/.jdks/openjdk-23.0.1
export PATH=$JAVA_HOME/bin:$PATH

# Use INFO level for a cleaner QR code display, but keep the opens for reflection
java --add-opens java.base/java.lang=ALL-UNNAMED \
     --add-opens java.base/java.lang.reflect=ALL-UNNAMED \
     -Dorg.slf4j.simpleLogger.defaultLogLevel=info \
     -jar target/whatsapp-service-1.0-SNAPSHOT.jar
