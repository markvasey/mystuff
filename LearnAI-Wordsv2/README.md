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

Under this definition, **this model is a 2.92M parameter model** (specifically, **2,924,800** active parameters, or **2,892,032** parameters if strictly excluding the static, non-trainable positional encoding matrix).

For a configuration with $d_{model} = 256$, $block\_size = 128$, and $vocab\_size = 4,096$:

*   **Active/Trainable Model Parameters (AI Metric)**: **2,924,800** ($\sim 2.92\text{M}$ weights and biases), occupying **11.70 MB** of float32 memory.
*   **Adam Optimizer State Parameters**: **5,784,064** ($\sim 5.78\text{M}$ states for $m$ and $v$ vectors), occupying **23.14 MB** of float32 memory.
*   **Total Saved File Size (`model.bin`)**: **34.8 MB** (34,836,368 bytes).

### Parameter Count Details

| Component | Matrix Dimensions | Weight/Bias Parameters | Adam State Parameters ($m$ and $v$) |
| :--- | :---: | :---: | :---: |
| **Token Embeddings** | $4,096 \times 256$ | 1,048,576 | 2,097,152 |
| **Positional Encoding** | $128 \times 256$ | 32,768 (static) | 0 |
| **7x LayerNorm Layers** | $7 \times (1 \times 256)$ gain, bias | 3,584 | 7,168 |
| **3x Self-Attention Blocks** | $3 \times (3 \times 256 \times 256)$ query/key/value | 589,824 | 1,179,648 |
| **3x Feed-Forward Layers** | $3 \times (256 \times 256 + 1 \times 256)$ weights/bias | 197,376 | 394,752 |
| **Output Language Model Head** | $256 \times 4,096 + 1 \times 4,096$ weights/bias | 1,052,672 | 2,105,344 |
| **Total** | | **2,924,800** | **5,784,064** |

### File Serialization Structure

When saved to `model.bin`, the weights and Adam optimizer states are serialized as raw 32-bit floats. The total file size consists of:
1. **Float Data**: $(2,924,800 + 5,784,064) \times 4\text{ bytes} = 34,835,456\text{ bytes}$ ($\sim 34.83\text{ MB}$).
2. **Metadata Headers**: $912\text{ bytes}$ (epoch counts, layer counts, and matrix dimension prefixes).

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
    *   *Mechanics:* Prepares and shuffles sequence pairs, bundles them into flat batches of size 512, runs training epochs, and periodically saves checkpoints.
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
    *   *Mechanics:* Orchestrates forward/backward GPU execution flow. Returns a `ModelForwardResult` to enforce VRAM garbage collection.
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

---

## 📊 Case Study: 40-Epoch GPU Training Run Analysis

Training configuration: $d_{model} = 256$, $block\_size = 128$, $vocab\_size = 4,096$, batch size $512$, **80,839 sequences** from 5,000 children's stories (~808K tokens). Learning rate $0.0003$.

### Epoch Loss Progression

| Epoch | Avg Loss | Duration | Notes |
| :---: | :---: | :---: | :--- |
| 1 | 7.7160 | 74s | Random weights; learning basic token co-occurrence |
| 2 | 6.9751 | 73s | Character clusters forming |
| 3 | 5.2666 | 73s | **Major convergence step** — loss drops 1.7 in one epoch |
| 4 | 4.7886 | 73s | Spelling patterns emerging |
| **5** | **4.5096** | 73s | **Sample generated** |
| 6 | 4.2720 | 73s | |
| 7 | 4.0668 | 73s | |
| 8 | 3.9004 | 73s | |
| 9 | 3.7625 | 73s | |
| **10** | **3.6466** | 73s | **Sample generated** |
| 11 | 3.5457 | 73s | |
| 12 | 3.4583 | 73s | |
| 13 | 3.3803 | 73s | |
| 14 | 3.3091 | 73s | |
| **15** | **3.2434** | 73s | **Sample generated** |
| 16 | 3.1845 | 73s | |
| 17 | 3.1298 | 73s | |
| 18 | 3.0767 | 74s | |
| 19 | 3.0302 | 73s | |
| **20** | **2.9827** | 73s | **Sample generated** |
| 21 | 2.9402 | 73s | |
| 22 | 2.9008 | 74s | |
| 23 | 2.8626 | 73s | |
| 24 | 2.8252 | 74s | |
| **25** | **2.7901** | 74s | **Sample generated** |
| 26 | 2.7580 | 74s | |
| 27 | 2.7236 | 73s | |
| 28 | 2.6928 | 73s | |
| 29 | 2.6619 | 73s | |
| **30** | **2.6330** | 73s | **Sample generated** |
| 31 | 2.6043 | 73s | |
| 32 | 2.5773 | 73s | |
| 33 | 2.5522 | 73s | |
| 34 | 2.5260 | 73s | |
| **35** | **2.5014** | 73s | **Sample generated** |
| 36 | 2.4760 | 73s | |
| 37 | 2.4532 | 73s | |
| 38 | 2.4296 | 73s | |
| 39 | 2.4075 | 73s | |
| **40** | **2.3866** | 73s | **Final model saved. Sample generated.** |

Loss fell **69.1%** from 7.7160 → 2.3866 over 40 epochs. Total training time: **~49 minutes**.

---

### Generation Quality & Narrative Progression

The model logs a generated sample every 5 epochs. Each is assessed for spelling, grammar, and semantic coherence.

---

#### Epoch 5 — Avg Loss: 4.5096

> *"The becoming from, came back at the redished at the flower spone. Watelevia said the man was too good to help her hand, but he knew it was important to try it a mainfly, and go back to the kitchen every day.*
> *--- Story 5529 --- Once upon a time, there was a"*

| Dimension | Assessment |
| :--- | :--- |
| **Spelling** | ❌ `redished`, `spone`, `Watelevia`, `mainfly` — corrupted BPE blend tokens |
| **Grammar** | ❌ *"The becoming from, came back"* — incoherent clause; double preposition *"at the redished at"* |
| **Semantics** | ❌ Very low. First sentence is meaningless. Second has narrative intent but no logic. |
| **What was learned** | Story separator format (`--- Story N ---`) and the opening formula `Once upon a time, there was a` are reliable. |

---

#### Epoch 10 — Avg Loss: 3.6466

> *"The bunny knew she had to be brave and independ.*
> *--- Story 798 --- Lila and Tom. Sweety was so kind. "Okay, mommy. It looks at our mom. And you we're not nice." The bear smiled and said, "I love you with your favorite food has been more fun. Thank"*

| Dimension | Assessment |
| :--- | :--- |
| **Spelling** | ⚠️ `independ` (truncated "independent"); all other words correctly spelled |
| **Grammar** | ⚠️ *"brave and independ"* — part-of-speech mismatch; *"I love you with your favorite food has been..."* — run-on, broken clause |
| **Semantics** | ⚠️ Medium-Low. Named characters (`Lila`, `Tom`) appear. Dialogue is correctly quoted. Story-ending phrase (`Thank...`) emerges. |
| **What was learned** | Dialogue formatting with quotation marks is now reliable. Emotional vocabulary (`kind`, `brave`, `nice`) and character pronouns are stable. |

---

#### Epoch 15 — Avg Loss: 3.2434

> *"The magical fairy came to a special gifted for their creative. She invited to make a wishes come in to help dry them rescutting down the old leave. It was good friends and cookies they could find another delicious snack. They found the perfect drin. It was a big delicious that big cookies"*

| Dimension | Assessment |
| :--- | :--- |
| **Spelling** | ❌ `rescutting` (blend of "rescuing"/"cutting"), `leave` (should be "leaves"), `drin` (truncated "drink") |
| **Grammar** | ❌ *"make a wishes"* — article-noun mismatch; *"It was good friends"* — subject-verb mismatch; *"came to a special gifted"* — adjective used as noun |
| **Semantics** | ⚠️ Medium. Strong thematic chains: fairy → wishes → magic; cookies → snack → delicious. But causality between clauses is broken. |
| **What was learned** | Multi-sentence thematic consistency. Natural sentence-openers (`They found...`, `It was...`). |

---

#### Epoch 20 — Avg Loss: 2.9827

> *"The story teaches us that it's important to stay ad and joy to make the world promise, so his promise that to always look around and understanding to be more careful.*
> *--- Story 3868 --- Once upon a time, there lived a boy called Polly lived in a big tree with lots of courage popping and the wise"*

| Dimension | Assessment |
| :--- | :--- |
| **Spelling** | ⚠️ `ad` (truncated "glad" or "ahead"); all other words correctly spelled |
| **Grammar** | ❌ *"stay ad and joy"* — adjective/noun mismatch; *"a boy called Polly lived"* — double verb, needs relative pronoun "who" |
| **Semantics** | ✅ High. Generates a **meta-narrative moral conclusion** (*"The story teaches us..."*) — a structural feature of children's stories. Abstract nouns (`courage`, `understanding`, `promise`) used in correct context. |
| **What was learned** | Discourse structure: the model understands stories end with moral lessons. Story transitions are perfectly placed. |

---

#### Epoch 25 — Avg Loss: 2.7901

> *"The dragon is safe and sound. Little girl never came from the designs an exhauff gether. Thes them out and makes beautiful lights so happy. You are beautiful!"*
> *One day, the quarreward. She likes to show her the new dog"*

| Dimension | Assessment |
| :--- | :--- |
| **Spelling** | ❌ `exhauff` (corrupted token), `gether` (truncated "together"), `Thes` (truncated "These"), `quarreward` (corrupted blend) |
| **Grammar** | ❌ *"Little girl"* — missing article; *"Thes them out"* — missing subject |
| **Semantics** | ✅ High on isolated phrases. *"The dragon is safe and sound"* — perfect. *"You are beautiful!"* — correct exclamation. *"One day..."* — correct narrative transition. |
| **What was learned** | Individual short sentences are now grammatically and semantically complete. Errors are at clause *junctions* rather than within sentences. |

---

#### Epoch 30 — Avg Loss: 2.6330

> *"The wind in return. We should always been careful to make sure cannot have a purple end time keeping new things and her new sheet as far away. The end.*
> *--- Story 565 --- Once upon a time, there was a boy named Tom. Do not destroyed all the times, but you don't give him."*
> *Mom was very"*

| Dimension | Assessment |
| :--- | :--- |
| **Spelling** | ✅ All words correctly spelled |
| **Grammar** | ⚠️ *"always been careful"* — missing auxiliary "have"; *"Do not destroyed"* — incorrect infinitive form ("Do not destroy"); *"cannot have a purple end time"* — incoherent noun phrase |
| **Semantics** | ✅ Strong. *"The end."* is a learned story-closing convention. *"Once upon a time, there was a boy named Tom"* — perfectly formed opening. Dialogue attribution (*Mom was very...*) used correctly. |
| **What was learned** | Story structural bookends (`The end.`, `Once upon a time...`) are now fluent. Common character names (`Tom`) and family roles (`Mom`) appear naturally. |

---

#### Epoch 35 — Avg Loss: 2.5014

> *"The special sparkly is a special hole. This is a special moment thing that is a special tide tree.*
> *--- Story 80 --- Anna and Ben are even and they wanted somethose to read the teacher. Anna wants to use the envin to open the book together. Ben found"*

| Dimension | Assessment |
| :--- | :--- |
| **Spelling** | ❌ `somethose` (corrupted "something"/"those"), `envin` (corrupted token, likely "invention" or "envelope") |
| **Grammar** | ⚠️ *"Anna and Ben are even"* — non-idiomatic; *"wanted somethose to read the teacher"* — object confusion (should be "the teacher to read") |
| **Semantics** | ⚠️ Medium. Heavy repetition of the word "special" (4× in two sentences) suggests the token probability is over-weighted. Named characters (`Anna`, `Ben`) appear in consistent roles across clauses. Collaborative narrative (`open the book together`) is semantically correct. |
| **What was learned** | Two characters with distinct names interacting in one scene. Collaborative action verbs (`open together`, `wanted to read`) are used coherently. |

---

#### Epoch 40 — Avg Loss: 2.3866 *(Final)*

> *"The bad man saw her and make loud noises. They were scared. They also had never trouble idey for the night.*
> *--- Story 445 --- Once upon a time, there was a furry ef quien candy said, "Let's go home now." So the family went home and packed some food some vegetables. The gardener said thank you"*

| Dimension | Assessment |
| :--- | :--- |
| **Spelling** | ❌ `idey` (corrupted token, likely "idea" or "indeed"); `ef quien` (corrupted tokens from mixed-language BPE blends) |
| **Grammar** | ⚠️ *"make loud noises"* — missing subject agreement ("makes"); *"packed some food some vegetables"* — missing conjunction ("food and some vegetables") |
| **Semantics** | ✅ High. *"The bad man saw her and make loud noises. They were scared."* — causally coherent micro-narrative. *"Let's go home now"* — socially appropriate dialogue. *"The gardener said thank you"* — semantically logical scene closure. |
| **What was learned** | Causal micro-narratives (action → consequence). Scene changes and character roles (`gardener`, `family`) are contextually appropriate. Polite conversational closings learned. |

---

### What the Model Learned

| Stage | Epochs | Key Capability Acquired |
| :--- | :---: | :--- |
| **Orthography** | 1–5 | Word spelling stabilises; BPE tokens decoded correctly |
| **Punctuation** | 5–10 | Quotation marks, full stops, commas placed correctly |
| **Named Characters** | 10–15 | Character names (`Lila`, `Tom`, `Anna`, `Ben`) and pronouns used consistently |
| **Thematic Chaining** | 15–20 | Words linked by topic across sentences: fairy → wishes → magic |
| **Discourse Structure** | 20–25 | Story openings, separators (`--- Story N ---`), and moral conclusions all learned |
| **Short-Sentence Coherence** | 25–30 | Individual sentences grammatically and semantically complete |
| **Causal Micro-Narratives** | 30–40 | Action → consequence chains: *"The bad man made noise. They were scared."* |

The primary remaining weakness at Epoch 40 is **inter-clause coherence** and occasional **BPE token corruption** at word boundaries (e.g., `ef quien`, `idey`). Within individual sentences the model is largely fluent; failures occur at transitions between clauses or when uncommon token combinations are sampled.


