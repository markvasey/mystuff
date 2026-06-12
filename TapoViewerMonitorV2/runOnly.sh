#!/bin/bash

# Exit on error
set -e

echo "=== TapoViewerMonitor Fast Launcher (exec:exec) ==="

# 1. Set JDK 26 Environment
export JAVA_HOME="/home/markvasey/.sdkman/candidates/java/26.0.1-tem"
if [ ! -d "$JAVA_HOME" ]; then
    echo "ERROR: JDK 26 not found at $JAVA_HOME"
    exit 1
fi
export PATH="$JAVA_HOME/bin:$PATH"

# 2. Set CUDA library paths for system76 cudnn and cuda-11.2
export LD_LIBRARY_PATH="/usr/lib/cuda-11.2/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"

# 3. Check for native libseizure_cuda.so (compile if missing)
if [ ! -f "libseizure_cuda.so" ]; then
    echo "Native CUDA library 'libseizure_cuda.so' not found. Compiling now..."
    make -C src/main/native
fi

# 4. Ensure target directory exists and set Dropbox ignore attribute
mkdir -p target
python3 -c "import os; os.setxattr('target', 'user.com.dropbox.ignored', b'1')" 2>/dev/null || true

# 5. Launch Application using Maven exec:exec@run-exec (Incremental & Fast)
echo "Launching application..."
exec ./mvnw exec:exec@run-exec "$@"
