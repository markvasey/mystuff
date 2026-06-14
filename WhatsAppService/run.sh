#!/bin/bash
# Get the absolute path to the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export JAVA_HOME=/home/markvasey/.sdkman/candidates/java/26.0.1-tem
export PATH=$JAVA_HOME/bin:$PATH

# Run using the absolute path to the jar
java --add-opens java.base/java.lang=ALL-UNNAMED \
     --add-opens java.base/java.lang.reflect=ALL-UNNAMED \
     -Dorg.slf4j.simpleLogger.defaultLogLevel=debug \
     -jar "$SCRIPT_DIR/target/whatsapp-service-1.0-SNAPSHOT.jar"
