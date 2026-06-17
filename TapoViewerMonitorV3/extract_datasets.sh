#!/bin/bash
# Exit on error
set -e

echo "=== Initializing Java 26 Environment ==="
export JAVA_HOME="/home/markvasey/.sdkman/candidates/java/26.0.1-tem"
if [ ! -d "$JAVA_HOME" ]; then
    echo "ERROR: JDK 26 not found at $JAVA_HOME"
    exit 1
fi
export PATH="$JAVA_HOME/bin:$PATH"

# Set CUDA library path for GPU-accelerated preprocessing in YOLOv8-pose
export LD_LIBRARY_PATH="/usr/lib/cuda-11.2/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"

echo "=== Compiling Project (Test Scope) ==="
./mvnw test-compile

echo "=== 1. Extracting Training Seizures (Label: 1) ==="
./mvnw exec:java -Dexec.classpathScope=test \
  -Dexec.mainClass=com.tapoviewer.cli.DatasetExtractor \
  -Dexec.args="TestVideos/Training_Calibration_Seizures 1"

echo "=== 2. Extracting Training Non-Seizures (Label: 0) ==="
./mvnw exec:java -Dexec.classpathScope=test \
  -Dexec.mainClass=com.tapoviewer.cli.DatasetExtractor \
  -Dexec.args="TestVideos/Training_Calibration_NonSeizures 0"

echo "=== 3. Extracting Evaluation Seizures (Label: 1) ==="
./mvnw exec:java -Dexec.classpathScope=test \
  -Dexec.mainClass=com.tapoviewer.cli.DatasetExtractor \
  -Dexec.args="TestVideos/Evaluation_Seizures 1"

echo "=== 4. Extracting Evaluation Non-Seizures (Label: 0) ==="
./mvnw exec:java -Dexec.classpathScope=test \
  -Dexec.mainClass=com.tapoviewer.cli.DatasetExtractor \
  -Dexec.args="TestVideos/Evaluation_NonSeizures 0"

echo "=== All Datasets Successfully Extracted! ==="
