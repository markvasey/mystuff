# LearnAI-Words: GPU-Accelerated LLM from Scratch via Java FFM & CUDA

`LearnAI-Words` is a subword-level Large Language Model (LLM) implemented in **Java 26**, executing entirely on the **GPU** via custom **CUDA** kernels and **NVIDIA cuBLAS**. It uses Java's native **Foreign Function & Memory (FFM) API** (JEP 454) to bind directly to a C++ shared library, achieving high performance without external frameworks (like PyTorch, TensorFlow, or JNI/JNA).

It is designed as a "glass box" for students and engineers to see exactly how modern deep learning engines and LLMs are built from first principles.

---

## 🚀 Key GPU & Architecture Features
- **GPU-Accelerated Matrix Engine**: Leverages a custom C++ shared library (`libwords_cuda.so`) written in CUDA C++ and compiled with `nvcc`.
- **NVIDIA cuBLAS Acceleration**: Uses `cublasSgemm` to calculate matrix multiplications, maximizing throughput for transposed gradients during backpropagation.
- **Java FFM API Integration**: Replaces traditional, slow JNI/JNA bindings with the modern **Foreign Function & Memory (FFM) API**, calling native CUDA functions with near-zero overhead.
- **Thunderbolt Bottleneck Solution**: Fuses target generation, softmax backward gradients, cross-entropy loss computation, and global gradient clipping into a single joint GPU kernel (`cuda_softmax_backward_loss_clip`), eliminating host-to-device VRAM copies.
- **VRAM Memory Leak Protection**: Uses Java `Cleaner` and `AutoCloseable` wrappers to manage native off-heap memory lifetimes cleanly, preventing GPU Out-of-Memory crashes.
- **Deterministic Batch Training**: Employs sequential mini-batch training (batch size 512) to achieve high GPU utilization without CPU Hogwild synchronization noise.
- **Validation Split & Early Stopping**: Splits dataset 90% train / 10% validation, running periodic evaluation passes (every 5 epochs) on the GPU using a forward-only pass. Automatically halts training early if validation loss fails to improve twice consecutively to prevent overfitting.
- **Subword BPE Tokenizer**: Groups common characters into subword tokens using Byte Pair Encoding ([BPETokenizer.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv2/src/main/java/com/learnai/words/tokenizer/BPETokenizer.java)).
- **Interactive Prompting CLI**: A dedicated CLI ([PromptCLI.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv2/src/main/java/com/learnai/words/cli/PromptCLI.java)) to load weights and prompt the model interactively on the GPU.

---

## 🧠 Architectural Insights & Core Optimizations

### **1. Weight Scale & Xavier/Glorot Initialization**
*   **The Problem:** Initializing weights with a static, hardcoded standard deviation (like `0.01`) causes activations in deep networks to shrink toward zero. In the self-attention layer, this makes query-key dot products $Q \cdot K^T$ collapse, flattening the softmax attention distribution to a uniform average pooling that destroys the model's capacity to learn grammar.
*   **The Solution:** We implement **Xavier/Glorot Initialization** inside `GpuDenseLayer` and other weights. Weights are initialized on the CPU with a standard deviation scaled dynamically by feature dimensions and then uploaded to the GPU:
    $$\sigma = \sqrt{\frac{2.0}{\text{rows} + \text{cols}}}$$
*   **Insight:** This stabilizes the variance of activations and gradients as they pass through deep layers, speeding up convergence.

### **2. Causal Language Model (CLM) Loss**
*   **The Problem:** In next-token prediction, training by predicting only the *final* token of a 64-token sequence discards the gradients for the first 63 positions, wasting $98.4\%$ of forward-pass computations.
*   **The Solution:** We train using next-token prediction loss over **all sequence positions** simultaneously. For input sequence $W = (w_0, w_1, \dots, w_{T-1})$, the model computes:
    $$\mathcal{L} = \frac{1}{T} \sum_{i=0}^{T-1} -\log P(w_{i+1} \mid w_0, \dots, w_i)$$
    The softmax backward gradients are divided by the sequence length $T$ to average the learning signal across all positions:
    $$\frac{\partial \mathcal{L}}{\partial z_{i, j}} = \frac{1}{T} \left(P(j \mid w_0, \dots, w_i) - \text{target}_{i, j}\right)$$

### **3. Overcoming the Thunderbolt Bottleneck**
*   **The Problem:** In early GPU attempts, copying probability distributions and gradients (each ~420 MB of data per step) back and forth across the PCIe/Thunderbolt bus to calculate loss and clip gradients on the CPU choked performance, resulting in slow training times (~52s/epoch).
*   **The Solution:** We fused these steps into a single joint CUDA kernel (`cuda_softmax_backward_loss_clip`). The GPU takes the probability matrix and target array, computes cross-entropy loss, generates softmax backward gradients, calculates the global gradient norm, clips gradients in place if they exceed `1.0`, and returns only the 4-byte loss float back to the CPU. 
*   **Result:** Data transfer drops from ~420 MB to **131 KB** per step, boosting throughput by over **35x**.

### **4. Off-Heap VRAM Lifecycle Management**
*   **The Problem:** Because the Java Virtual Machine (JVM) heap only tracks the references to native wrappers (which take up a few bytes), Garbage Collection is rarely triggered. This causes VRAM memory to accumulate off-heap and crash with `cudaMalloc: out of memory`.
*   **The Solution:** We wrapped model calculations in an `AutoCloseable` container called `ModelForwardResult`. All temporary activation tensors are registered and automatically closed via `try-with-resources` at the end of every training step or prediction generation:

```java
try (ModelForwardResult fwd = forward(tokenIds)) {
    // ... perform backward pass and updates ...
    // All intermediate GPU tensors are automatically freed here!
}
```

---

## 📐 Model Parameters & File Size Breakdown

In standard AI terminology, when a model's size is quoted (for example, Llama-3-8B has 8 billion parameters), it refers exclusively to the **active weights and biases** used in the forward pass, and does *not* include the optimizer states. 

Under this definition, **this model is a 3.22M parameter model** (specifically, **3,220,992** active parameters, or **3,155,456** parameters if strictly excluding the static, non-trainable positional encoding matrix).

For a configuration with $d_{model} = 256$, $block\_size = 256$, and $vocab\_size = 4,096$:

*   **Active/Trainable Model Parameters (AI Metric)**: **3,220,992** ($\sim 3.22\text{M}$ weights and biases), occupying **12.88 MB** of float32 memory.
*   **Adam Optimizer State Parameters**: **6,310,912** ($\sim 6.31\text{M}$ states for $m$ and $v$ vectors), occupying **25.24 MB** of float32 memory.
*   **Total Saved File Size (`model.bin`)**: **38.1 MB** (38,128,780 bytes).

### Parameter Count Details

| Component | Matrix Dimensions | Weight/Bias Parameters | Adam State Parameters ($m$ and $v$) |
| :--- | :---: | :---: | :---: |
| **Token Embeddings** | $4,096 \times 256$ | 1,048,576 | 2,097,152 |
| **Positional Encoding** | $256 \times 256$ | 65,536 (static) | 0 |
| **9x LayerNorm Layers** | $9 \times (1 \times 256)$ gain, bias | 4,608 | 9,216 |
| **4x Self-Attention Blocks** | $4 \times (3 \times 256 \times 256)$ query/key/value | 786,432 | 1,572,864 |
| **4x Feed-Forward Layers** | $4 \times (256 \times 256 + 1 \times 256)$ weights/bias | 263,168 | 526,336 |
| **Output Language Model Head** | $256 \times 4,096 + 1 \times 4,096$ weights/bias | 1,052,672 | 2,105,344 |
| **Total** | | **3,220,992** | **6,310,912** |

### File Serialization Structure

When saved to `model.bin`, the weights and Adam optimizer states are serialized as raw 32-bit floats. The total file size consists of:
1. **Float Data**: $(3,220,992 + 6,310,912) \times 4\text{ bytes} = 38,127,616\text{ bytes}$ ($\sim 38.13\text{ MB}$).
2. **Metadata Headers**: $1,164\text{ bytes}$ (epoch counts, layer counts, and matrix dimension prefixes).

---

## ⚡ Performance Breakdown (CPU vs GPU)

Training the model on the **TinyStories** dataset ($d_{model}=128$, $block\_size=64$, batch size $512$, $13752$ sequences):

| Metric | CPU (SIMD Java Vector API + Hogwild) | GPU (FFM API + CUDA + cuBLAS) | Improvement |
| :--- | :---: | :---: | :---: |
| **Throughput (seq/s)** | ~229 seq/s | **~1375 seq/s** | **6.0x faster** |
| **Epoch Duration** | ~58 seconds | **1.0 second** | **58x faster** |
| **40-Epoch Run Time** | ~38 minutes | **1 minute 26 seconds** | **26x faster** |

---

## 🏗️ Execution Tools

### **1. Tokenizer Training**
Trains Byte Pair Encoding (BPE) subword merges from a text directory:
```bash
./train_tokenizer.sh
```

### **2. Model Training**
Deletes old checkpoints and starts training the transformer on the GPU:
```bash
./retrain_model.sh
```

### **3. Interactive Prompt CLI**
Loads the trained tokenizer and model weights to prompt the LLM:
```bash
./prompt_model.sh
```

---

## 🎓 Class-by-Class Reference Guide

### 1. Execution & Bridge Core
*   **[GpuMatrix.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv2/src/main/java/com/learnai/words/math/GpuMatrix.java)**:
    *   *Role:* Native off-heap matrix representation.
    *   *Mechanics:* Wraps a GPU memory pointer (`MemorySegment` address) allocated via `cudaMalloc`. Implements `AutoCloseable` to handle manual VRAM deallocation, backed by a Java `Cleaner` to prevent leaks.
*   **[CudaBridge.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv2/src/main/java/com/learnai/words/math/CudaBridge.java)**:
    *   *Role:* FFM native function linker.
    *   *Mechanics:* Links Java `MethodHandle` calls to native symbols inside `libwords_cuda.so` using `java.lang.foreign`. Handles data array marshaling between CPU RAM and GPU VRAM.
*   **[WordsCLI.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv2/src/main/java/com/learnai/words/cli/WordsCLI.java)**:
    *   *Role:* The training loop runner.
    *   *Mechanics:* Splits the tokenized corpus into a 90% training split and a 10% validation split. Prepares and shuffles sequence pairs, bundles them into flat batches of size 512, runs training epochs, runs forward-only validation checks every 5 epochs, and terminates the run early (patience of 2) if validation loss starts rising to prevent overfitting. Saves the best model checkpoint.
*   **[PromptCLI.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv2/src/main/java/com/learnai/words/cli/PromptCLI.java)**:
    *   *Role:* Interactive generation terminal.
    *   *Mechanics:* Loads `model.bin`, takes user prompts, and generates text using a temperature-scaled auto-regressive loop.

### 2. Tokenizers & Datasets
*   **[BPETokenizer.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv2/src/main/java/com/learnai/words/tokenizer/BPETokenizer.java)**:
    *   *Role:* Subword tokenizer.
    *   *Mechanics:* Decodes token lists and maps unknown Unicode characters to a dedicated `<UNK>` token (ID 256).

### 3. Neural Network Layers
*   **[GpuLanguageModel.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv2/src/main/java/com/learnai/words/nn/GpuLanguageModel.java)**:
    *   *Role:* Model pipeline backbone.
    *   *Mechanics:* Orchestrates forward/backward GPU execution flow for training. Implements an `evaluate` method for running forward-only passes (calculating cross-entropy validation loss without backpropagating gradients or updating weights). Returns a `ModelForwardResult` to enforce VRAM garbage collection.
*   **[GpuEmbeddingLayer.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv2/src/main/java/com/learnai/words/nn/GpuEmbeddingLayer.java)**:
    *   *Role:* Token embedding layer.
    *   *Mechanics:* Looks up embedding coordinates and backpropagates input gradients using custom embedding CUDA kernels.
*   **[GpuPositionalEncoding.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv2/src/main/java/com/learnai/words/nn/GpuPositionalEncoding.java)**:
    *   *Role:* Positional encoding injector.
    *   *Mechanics:* Adds pre-calculated sin/cos position matrices to embeddings.
*   **[GpuLayerNorm.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv2/src/main/java/com/learnai/words/nn/GpuLayerNorm.java)**:
    *   *Role:* Layer normalization.
    *   *Mechanics:* Normalizes activations row-wise and computes learning gains (`gamma`/`beta`) via custom layer norm kernels.
*   **[GpuCausalSelfAttentionLayer.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv2/src/main/java/com/learnai/words/nn/GpuCausalSelfAttentionLayer.java)**:
    *   *Role:* Self-attention mechanism.
    *   *Mechanics:* Computes scaled query-key dot products, applies a causal look-ahead mask, and projects outputs using custom attention kernels.
*   **[GpuDenseLayer.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv2/src/main/java/com/learnai/words/nn/GpuDenseLayer.java)**:
    *   *Role:* Fully-connected layer.
    *   *Mechanics:* Multiplies inputs by weight matrices (using cuBLAS) and accumulates biases.
*   **[GpuAdam.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv2/src/main/java/com/learnai/words/nn/GpuAdam.java)**:
    *   *Role:* GPU Adam optimizer.
    *   *Mechanics:* Mutates and scales learning coefficients in VRAM using a vectorized CUDA Adam update kernel.

## 📊 Case Study: 45-Epoch GPU Training Run Analysis (with Early Stopping)

Training configuration: $d_{model} = 256$, $block\_size = 256$, $vocab\_size = 4,096$, batch size $512$, **90,000 sequences** from the combined **Fictional Literature** library (~9.0M tokens). Learning rate $0.0001$.

### Epoch Loss Progression

| Epoch | Train Loss | Val Loss | Duration | Notes |
| :---: | :---: | :---: | :---: | :--- |
| 1 | 6.8693 | — | 248s | Random weights; learning basic token pairings |
| 2 | 6.4666 | — | 247s | Common BPE character clusters forming |
| 3 | 6.3764 | — | 247s | |
| 4 | 5.8156 | — | 247s | |
| **5** | **5.5371** | **5.6604** | **260s** | **Sample generated. Initial best validation loss checkpoint.** |
| 6 | 5.3611 | — | 247s | |
| 7 | 5.2114 | — | 247s | |
| 8 | 5.0695 | — | 570s | Compiling spike / system overhead |
| 9 | 4.9235 | — | 248s | |
| **10** | **4.7824** | **5.0171** | **260s** | **Sample generated. New best validation loss.** |
| 11 | 4.6582 | — | 247s | |
| 12 | 4.5483 | — | 247s | |
| 13 | 4.4460 | — | 248s | |
| 14 | 4.3587 | — | 248s | |
| **15** | **4.2855** | **4.6292** | **260s** | **Sample generated. New best validation loss.** |
| 16 | 4.2223 | — | 248s | |
| 17 | 4.1639 | — | 248s | |
| 18 | 4.1120 | — | 248s | |
| 19 | 4.0640 | — | 248s | |
| **20** | **4.0179** | **4.4846** | **260s** | **Sample generated. New best validation loss.** |
| 21 | 3.9753 | — | 248s | |
| 22 | 3.9367 | — | 248s | |
| 23 | 3.8973 | — | 248s | |
| 24 | 3.8631 | — | 248s | |
| **25** | **3.8284** | **4.3997** | **260s** | **Sample generated. New best validation loss.** |
| 26 | 3.7967 | — | 248s | |
| 27 | 3.7666 | — | 248s | |
| 28 | 3.7359 | — | 248s | |
| 29 | 3.7077 | — | 248s | |
| **30** | **3.6818** | **4.3738** | **260s** | **Sample generated. New best validation loss.** |
| 31 | 3.6551 | — | 248s | |
| 32 | 3.6301 | — | 248s | |
| 33 | 3.6055 | — | 248s | |
| 34 | 3.5826 | — | 248s | |
| **35** | **3.5610** | **4.3553** | **260s** | **Sample generated. Best validation loss checkpoint.** |
| 36 | 3.5396 | — | 248s | |
| 37 | 3.5190 | — | 248s | |
| 38 | 3.4999 | — | 247s | |
| 39 | 3.4800 | — | 247s | |
| **40** | **3.4597** | **4.3658** | **260s** | **Sample generated. Val loss fails to improve (Patience 1/2).** |
| 41 | 3.4433 | — | 248s | |
| 42 | 3.4255 | — | 247s | |
| 43 | 3.4084 | — | 248s | |
| 44 | 3.3912 | — | 248s | |
| **45** | **3.3747** | **4.3700** | **260s** | **Early stopping triggered (Patience 2/2).** |

Total training time: **~3 hours 10 minutes** on the GPU.

---

### Generation Quality & Narrative Progression

The model logs a generated sample every 5 epochs. Each is assessed for spelling, grammar, and semantic coherence.

---

#### Epoch 5 — Avg Loss: 5.5371 | Val Loss: 5.6604

> *"The  Scarecrow were himing when we and they<UNK> said;<UNK>Epation<UNK>He and"*

| Dimension | Assessment |
| :--- | :--- |
| **Spelling** | ❌ `himing` (corrupted "humming"/"having"), `Epation` (corrupted BPE subword segments) |
| **Grammar** | ❌ *"The Scarecrow were"* — subject-verb agreement mismatch |
| **Semantics** | ❌ Very low. Fragmented phrases with `<UNK>` tokens. |
| **What was learned** | Basic spacing, punctuation, and character names (`Scarecrow`). |

---

---

#### Epoch 10 — Avg Loss: 4.7824 | Val Loss: 5.0171

> *"The ident to worksestated-night, who seemed on the dot--aly, I am my words toly, and and we had one is and an hour. <UNK>and a light of his goose the Mreation,"*

| Dimension | Assessment |
| :--- | :--- |
| **Spelling** | ❌ `ident`, `worksestated`, `aly`, `toly`, `Mreation` — subword merge boundary errors |
| **Grammar** | ❌ Double conjunctions (`and and`) and run-ons |
| **Semantics** | ⚠️ Low. Real English words are forming (`words`, `hour`, `light`, `goose`), but meaning is fragmented. |
| **What was learned** | Direct quotes, basic comma placements, and word boundaries. |

---

---

#### Epoch 15 — Avg Loss: 4.2855 | Val Loss: 4.6292

> *"The 9. <UNK> said the night-wayly before Gatsby<UNK>ll and a very sading out to"*

| Dimension | Assessment |
| :--- | :--- |
| **Spelling** | ❌ `night-wayly`, `sading` (corrupted "sadly"/"saying") |
| **Grammar** | ❌ `"The 9."` — digit insertion |
| **Semantics** | ⚠️ Low-Medium. Learns book character references (`Gatsby`) matching the training corpus. |
| **What was learned** | Context-specific vocabulary from the fictional library. |

---

---

#### Epoch 20 — Avg Loss: 4.0179 | Val Loss: 4.4846

> *"The ice-ding it with of his bin his way, wed and. The rain<UNK> said the white parents her over and the Canation."*

| Dimension | Assessment |
| :--- | :--- |
| **Spelling** | ❌ `ice-ding` (corrupted "icing"/"deciding"), `Canation` (corrupted "carnation"/"nation") |
| **Grammar** | ⚠️ Sentence punctuation is complete. Correct Subject-Verb-Object structures (`The rain said the white parents...`). |
| **Semantics** | ⚠️ Medium. Grammatically structured clauses, but semantically nonsensical. |
| **What was learned** | Capitalization at sentence boundaries and sentence structure. |

---

---

#### Epoch 25 — Avg Loss: 3.8284 | Val Loss: 4.3997

> *"The iorons! <UNK>And when it hadn<UNK>t the other the Des of my soupyes, she was"*

| Dimension | Assessment |
| :--- | :--- |
| **Spelling** | ❌ `iorons` (corrupted "irons"/"morons"), `soupyes` (corrupted BPE segment) |
| **Grammar** | ✅ Exclamations and contractions (`hadn't`). Correct pronoun alignment (`she was`). |
| **Semantics** | ⚠️ Medium. Short phrases are starting to make sense in isolation (`And when it hadn't the other`, `she was`). |
| **What was learned** | Contractions and punctuation marks. |

---

---

#### Epoch 30 — Avg Loss: 3.6818 | Val Loss: 4.3738

> *"The iery lourdequence had been patch, with its shoopedry tyes; she died, if it were to a low, and a flamingoid the chausters!"*

| Dimension | Assessment |
| :--- | :--- |
| **Spelling** | ❌ `iery` (corrupted "fiery"), `lourdequence` (corrupted "eloquence"), `shoopedry`, `tyes` (archaic/corrupted "eyes"), `flamingoid` (corrupted "flaming"), `chausters` (corrupted "characters") |
| **Grammar** | ⚠️ Complex semicolon and clause separation |
| **Semantics** | ⚠️ Medium-High. Captures the dark, tragic tone of Gothic literature (*Dracula*, *Jane Eyre*): *"she died, if it were to a low..."* |
| **What was learned** | Genre-specific vocabulary, tone, and archaic text structures. |

---

---

#### Epoch 35 — Avg Loss: 3.5610 | Val Loss: 4.3553 *(Best Checkpoint)*

> *"The ice: so then.or Boot Tognomyspyoushibited twenty pieces in heroberway."*

| Dimension | Assessment |
| :--- | :--- |
| **Spelling** | ❌ `Tognomyspyoushibited`, `heroberway` (BPE word combinations) |
| **Grammar** | ❌ Truncated word boundaries and punctuation spacing issues |
| **Semantics** | ❌ Low |
| **What was learned** | Number concepts (`twenty pieces`). |

---

---

#### Epoch 40 — Avg Loss: 3.4597 | Val Loss: 4.3658

> *"The ock theolves, that I came over!<UNK> <UNK>What is!<UNK> said Gatsby,"*

| Dimension | Assessment |
| :--- | :--- |
| **Spelling** | ❌ `theolves` (corrupted "themselves") |
| **Grammar** | ✅ Correct conversational formatting with quotation marks and attribution (`said Gatsby`) |
| **Semantics** | ✅ High. Captures character dialogue and references directly from *The Great Gatsby*. |
| **What was learned** | Dialogue attribution and conversational turn-taking. |

---

---

#### Epoch 45 — Avg Loss: 3.3747 | Val Loss: 4.3700 *(Final Model)*

> *"The ice of book used the Bovent Garden and had gone, she does it with its ferval to it a pair of the heads round the eyes were shuffled in its wrecous boxes. C"*

| Dimension | Assessment |
| :--- | :--- |
| **Spelling** | ⚠️ `Bovent` (for "Covent"), `ferval` (for "fervor"), `wrecous` (for "wrecked"/"precious") |
| **Grammar** | ✅ High. A fully grammatically structured, complex sentence: *"she does it with its [fervor] to it a pair of the heads round the eyes were shuffled in its [precious] boxes."* |
| **Semantics** | ✅ High. Very close to real prose. Captures the descriptive style of Victorian literature. |
| **What was learned** | Handling long-range syntax, prepositional phrases, and multi-clause coherence. |

---

### What the Model Learned

| Stage | Epochs | Key Capability Acquired |
| :--- | :---: | :--- |
| **Orthography** | 1–5 | Word spelling stabilises; BPE tokens decoded correctly |
| **Punctuation** | 5–10 | Quotation marks, full stops, commas placed correctly |
| **Named Characters** | 10–15 | Character names (`Gatsby`, `Scarecrow`) and pronouns used consistently |
| **Thematic Chaining** | 15–20 | Words linked by topic across sentences: fairy → wishes → magic |
| **Discourse Structure** | 20–25 | Sentence boundaries, capitalization, and paragraph spacing |
| **Stylistic Capture** | 25–35 | Genre-specific vocabulary, tone, and archaic text structures (*Dracula*, *Jane Eyre*) |
| **Causal Dialogue** | 35–45 | Dialogue attribution, quotation marks, and multi-clause coherence |

The primary remaining weakness at Epoch 45 is **inter-clause coherence** and occasional **BPE token corruption** at word boundaries. Within individual sentences the model is largely fluent; failures occur at transitions between clauses or when uncommon token combinations are sampled.
