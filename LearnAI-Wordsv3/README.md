# LearnAI-Wordsv3: Hybrid PyTorch Training + ONNX Runtime Java Inference

`LearnAI-Wordsv3` represents the evolution of the `LearnAI-Words` project series, migrating from hand-crafted CPU and GPU/CUDA matrix algorithms to industry-standard deep learning libraries.

This project implements the **Hybrid Approach**:
1. **Training (Python + PyTorch)**: Build, train, and validate a standard Causal Transformer model (multi-head attention, MLP block with GELU activations, and LayerNorm) in PyTorch. The trained model is then exported directly to the standardized **ONNX (Open Neural Network Exchange)** format.
2. **Inference (Java + ONNX Runtime)**: Run the exported `.onnx` model using **ONNX Runtime Java** (`com.microsoft.onnxruntime`), completely eliminating manual off-heap VRAM cleaners, custom FFM/JNI bindings, and C++/CUDA compiler setups.

---

## 🚀 Key Advantages & Architectural Shifts

*   **Removal of Hand-Crafted C++/CUDA**: The `src/main/native` C++ code, JNI/FFM bindings (`CudaBridge.java`), and custom allocations (`GpuMatrix.java`) are replaced entirely by Microsoft's ONNX Runtime.
*   **Standardized Transformer Architecture**: The model is trained using standard PyTorch modules (`nn.MultiheadAttention`, `nn.LayerNorm`, and `nn.Linear`), allowing multi-head self-attention and non-linear MLP blocks (GELU) that were extremely complex to hand-code in CUDA.
*   **Dynamic Sequence Length**: The exported ONNX model uses dynamic axes, allowing the Java CLI to run inference on prompts of any sequence length (up to the trained `block_size`).
*   **Identical Tokenization**: BPE Tokenizer training and encoding logic is kept identical and compatible between Python ([tokenizer.py](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv3/tokenizer.py)) and Java ([BPETokenizer.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv3/src/main/java/com/learnai/words/tokenizer/BPETokenizer.java)), enabling zero-friction vocabulary serialization.

---

## 🏗️ Project Structure & Component Mappings

*   **[tokenizer.py](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv3/tokenizer.py)**: Python implementation of BPE Tokenizer. Can save/load in the exact binary format used by Java's `BPETokenizer`.
*   **[train.py](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv3/train.py)**: The PyTorch trainer. Sets up BPE tokenization, loads text files, trains the multi-layer Causal Transformer with cross-entropy loss and AdamW, and exports the final model to `model.onnx`.
*   **[OnnxLanguageModel.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv3/src/main/java/com/learnai/words/nn/OnnxLanguageModel.java)**: Loads `model.onnx` using the Java ONNX Runtime library. Feeds input tokens as a 2D tensor and retrieves output logits.
*   **[TextGenerator.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv3/src/main/java/com/learnai/words/nn/TextGenerator.java)**: Auto-regressive text generator executing temperature softmax and Top-K candidate sampling on the ONNX model output logits.
*   **[PromptCLI.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv3/src/main/java/com/learnai/words/cli/PromptCLI.java)**: Interactive terminal prompt client.

---

## ⚡ Setup & Execution Guide

### 1. Python Environment Setup
The python trainer requires `torch`, `onnx`, `onnxscript`, `numpy`, and `regex`. Build a local virtual environment:
```bash
python3 -m venv venv
venv/bin/pip install torch onnx numpy regex onnxscript
```

### 2. Model Training & ONNX Export
Run the training script on your corpus. By default, it reads text files from `Training/FictionalLiterature` and exports `model.onnx`:
```bash
./train_model.sh
```
*(Configurable parameters like batch size, block size, epochs, and dims can be adjusted inside the [train_model.sh](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv3/train_model.sh) script).*

### 3. Interactive Prompt CLI (Java)
Run the Java CLI to load `model.onnx` and interactively generate text:
```bash
./prompt_model.sh
```

### 4. Running Unit Tests
Compile and verify the test suite (which validates BPE tokenization and checks ONNX model inference if `model.onnx` is present):
```bash
./mvnw test
```
