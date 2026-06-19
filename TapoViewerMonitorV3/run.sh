#!/bin/bash

# Exit on error for script setup operations, but allow app run to exit normally
set -e

echo "=== TapoViewerMonitor eGPU-Accelerated Seizure Detector ==="

# 1. Set JDK 26 Environment
export JAVA_HOME="/home/markvasey/.sdkman/candidates/java/26.0.1-tem"
if [ ! -d "$JAVA_HOME" ]; then
    echo "ERROR: JDK 26 not found at $JAVA_HOME"
    exit 1
fi
export PATH="$JAVA_HOME/bin:$PATH"

echo "Using Java at: $(which java)"
java -version

# 2. Set CUDA library paths
# - pip-installed NVIDIA packages for CUDA 12 / cuDNN 9 support (including cufft, cublas, cudart, nvjitlink, cudnn)
# - lib/: project-local symlinks (libcufft.so.11, libcufftw.so.11) required by ORT 1.22.0 CUDA EP (pip version takes priority)
# - cuda-11.2: required for libseizure_cuda.so (custom NPP kernels) and YOLOv8 CUDA EP
# - ollama/cuda_v12: provides libcublasLt.so.12 required by ONNX Runtime 1.22.0 GPU EP
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.10/site-packages/nvidia/cufft/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/cublas/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$(pwd)/lib:/usr/local/lib/ollama/cuda_v12:/usr/lib/cuda-11.2/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
echo "LD_LIBRARY_PATH set (pip CUDA12/cuDNN9 + local lib + CUDA 11.2 + CUDA 12 cublas)."

# 3. Check NVIDIA Persistence Mode
if command -v nvidia-smi &> /dev/null; then
    PERSISTENCE_MODE=$(nvidia-smi -q | grep "Persistence Mode" | head -n 1 | awk -F: '{print $2}' | xargs)
    if [ "$PERSISTENCE_MODE" = "Disabled" ]; then
        echo "------------------------------------------------------------"
        echo "WARNING: NVIDIA driver Persistence Mode is Disabled."
        echo "This causes a ~40 second delay when initializing ONNX GPU session."
        echo "To enable persistence mode, run:"
        echo "  sudo nvidia-smi -pm 1"
        echo "------------------------------------------------------------"
    else
        echo "NVIDIA Persistence Mode: Enabled ($PERSISTENCE_MODE)"
    fi
else
    echo "WARNING: nvidia-smi command not found. Ensure NVIDIA driver is configured."
fi

# 4. Check for native libseizure_cuda.so
if [ ! -f "libseizure_cuda.so" ]; then
    echo "Native CUDA library 'libseizure_cuda.so' not found. Compiling now..."
    make -C src/main/native
fi

# 5. Ensure target directory exists and set Dropbox ignore attribute
mkdir -p target
python3 -c "import os; os.setxattr('target', 'user.com.dropbox.ignored', b'1')" 2>/dev/null || true

# 6. Package application to ensure latest changes are included
echo "Packaging application..."
./mvnw package -DskipTests

# 7. Execute Application
echo "Starting application..."
set +e
java --add-modules jdk.incubator.vector \
     --enable-native-access ALL-UNNAMED \
     -Djava.library.path=. \
     -jar target/TapoViewer.jar "$@"

exit $?
