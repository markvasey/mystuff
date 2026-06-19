#!/bin/bash

# 1. Set JDK 26 Environment
export JAVA_HOME="/home/markvasey/.sdkman/candidates/java/26.0.1-tem"
if [ ! -d "$JAVA_HOME" ]; then
    echo "ERROR: JDK 26 not found at $JAVA_HOME"
    exit 1
fi
export PATH="$JAVA_HOME/bin:$PATH"

# 2. Set CUDA library paths (prioritise pip-installed CUDA 12 / cuDNN 9 libraries)
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.10/site-packages/nvidia/cufft/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/cublas/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/usr/local/lib/ollama/cuda_v12:/usr/lib/cuda-11.2/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"

# 3. Compile and Execute Prompt CLI
./mvnw compile && ./mvnw exec:exec -Dexec.args="-Dblock.size=1024 -classpath %classpath com.learnai.words.cli.PromptCLI"

