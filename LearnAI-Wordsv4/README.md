# LearnAI-Wordsv4: Hybrid PyTorch Training + ONNX Runtime Java Inference

`LearnAI-Wordsv4` represents the evolution of the `LearnAI-Words` project series, migrating from hand-crafted CPU and GPU/CUDA matrix algorithms to industry-standard deep learning libraries, and now implementing modern NLP preprocessing and architecture standards.

This project implements the **Hybrid Approach**:
1. **Training (Python + PyTorch)**: Build, train, and validate a standard Causal Transformer model (multi-head attention, MLP block with SwiGLU activations, RoPE position encodings, GQA, and RMSNorm) in PyTorch. The trained model is then exported directly to the standardized **ONNX (Open Neural Network Exchange)** format.
2. **Inference (Java + ONNX Runtime)**: Run the exported `.onnx` model using **ONNX Runtime Java** (`com.microsoft.onnxruntime`), completely eliminating manual off-heap VRAM cleaners, custom FFM/JNI bindings, and C++/CUDA compiler setups.

---

## 🚀 Key Advantages & Architectural Shifts

*   **Removal of Hand-Crafted C++/CUDA**: The `src/main/native` C++ code, JNI/FFM bindings (`CudaBridge.java`), and custom allocations (`GpuMatrix.java`) are replaced entirely by Microsoft's ONNX Runtime.
*   **Native Hardware Delegation (cuBLAS & CUDA)**: Under the hood, the ONNX Runtime does not duplicate graphics or math libraries. Instead, it utilizes an **Execution Provider (EP)** architecture. On systems with an NVIDIA GPU, it automatically binds directly to NVIDIA's native proprietary libraries (such as **cuBLAS** for high-speed matrix multiplications and **cuDNN** for neural network modules) compiled directly in optimized C++ binaries. Java merely acts as an orchestrator, passing memory pointers via thin JNI wrappers so that the entire computation runs at raw hardware speeds without JVM memory or execution overhead.
*   **Standardized Transformer Architecture**: The model is trained using standard PyTorch modules, allowing multi-head self-attention and non-linear MLP blocks (SwiGLU) that were extremely complex to hand-code in CUDA.
*   **Dynamic Sequence Length**: The exported ONNX model uses dynamic axes, allowing the Java CLI to run inference on prompts of any sequence length (up to the trained `block_size`).
*   **Byte-Level Tokenization (BBPE)**: BPE Tokenizer training and encoding logic is kept identical and compatible between Python ([tokenizer.py](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv4/tokenizer.py)) and Java ([BPETokenizer.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv4/src/main/java/com/learnai/words/tokenizer/BPETokenizer.java)), working directly with raw UTF-8 bytes to achieve **0% unknown tokens (`<UNK>`)**.

---

## 🆕 LearnAI-Wordsv4 Enhancements: BBPE, Text Cleansing, SwiGLU, GQA, & RoPE

In `LearnAI-Wordsv4`, we address several major bottlenecks observed during the training and validation of `v3`: **Gutenberg metadata pollution** in prompts, **tokenization character corruption (`<UNK>`)** in outputs, and constraints on model scaling and attention memory.

### 1. Robust Text Preprocessing & Cleansing
To stop the model from learning Gutenberg index lines and page numbers (which caused output regressions like generating numbers or transcriber labels), we implemented a rigorous regex-based preprocessing pipeline inside `train.py`:
*   **Page Number Filtering:** Strips all standalone integers and `[Page X]` or `Page X` markings.
*   **Chapter & Act Headers:** Removes all lines containing Roman numerals or ordinal labels (e.g. `CHAPTER XXIV`, `ACT I`, `SCENE II`).
*   **Metadata & Transcriber Notes:** Filters out lines containing standard ebook labels (e.g., `Transcriber's Note:`, `Produced by...`).
*   **Standard Narrative Preservation:** Safely preserves instances where digits or chapter words appear inside actual narrative sentences (e.g. *"She was twelve years old."*).

### 2. Byte-Level Byte Pair Encoding (BBPE) Tokenizer
In `v3`, the character-based BPE mapping turned non-ASCII characters (like Gutenberg's curly quotes `“`/`”` and curly apostrophes `’`) into `<UNK>`. This polluted the generated dialogue with unreadable tags. 

We rewrote both the Python (`tokenizer.py`) and Java (`BPETokenizer.java`) engines to use **Byte-Level BPE (BBPE)**:
*   **Base Vocabulary (0–255):** Rather than characters, the base vocab is initialized with the 256 possible unsigned byte values.
*   **0% Unknown Tokens:** Every string is processed as raw UTF-8 bytes. Since all unicode characters are represented by byte combinations, `<UNK>` is completely eliminated from the vocabulary.
*   **Safe Decoding Stream:** When converting IDs back to text, the decoder concatenates the raw bytes first and decodes the final byte array to a UTF-8 string at the very end. This ensures multi-byte UTF-8 boundaries are resolved correctly without character corruption.

### 3. SwiGLU Gated Activation Function
In `v4`, we replaced the standard MLP activation (`GELU`) with **SwiGLU** (Swish Gated Linear Unit), which has become the standard in modern LLMs like LLaMA, PaLM, and Gemini. 
*   Instead of a simple projection and static threshold, the feed-forward layer projects input tokens into parallel gate and value streams, applies the Swish activation function, multiplies them element-wise, and projects the result back down.
*   This dynamic gating capability allows the model to learn complex logic gates and contextual associations with much smoother gradients, yielding faster convergence during training.

### 4. Rotary Position Embeddings (RoPE)
Instead of adding absolute position vectors (`wpe`) to the token vectors at the input layer, **RoPE** rotates the Query ($Q$) and Key ($K$) vectors in the complex plane during the attention calculation.
*   **Absolute Position Embeddings (Old Way):** Maps coordinates to absolute slots (`0` to `255`). This imposes a hard limit on sequence lengths—the model cannot process tokens past its trained limit without crashing or outputting gibberish.
*   **RoPE (Modern Standard):** Natively tracks the **relative distance** between words rather than their absolute positions. The rotation angle math extracts how far apart words are. This allows the model to extrapolate and generate text past its training limit (e.g., executing 2048+ tokens on a model trained for 1024) with minimal loss in quality.

### 5. Grouped-Query Attention (GQA)
Instead of having an equal number of Query ($Q$), Key ($K$), and Value ($V$) heads, GQA groups multiple Query heads to share single Key and Value heads.
*   **Multi-Head Attention (MHA - Old Way):** Every Query head has its own Key and Value heads (e.g., 8 Q, 8 K, 8 V). This forces the model to store a massive **KV Cache** in VRAM during text generation to avoid recalculations, slowing down generation speed on longer contexts.
*   **Grouped-Query Attention (GQA - Modern Standard):** Query heads are grouped. For example, 8 Query heads are split into 2 groups of 4, with each group sharing a single Key/Value head (8 Q, 2 K, 2 V). This cuts the KV Cache size by **4x**, dramatically reducing the VRAM footprint and speeding up Java ONNX text generation, without losing the modeling capacity of MHA.

---

## 🛠️ Optimization Tweaks & Training Best Practices

In addition to structural layout improvements, `v4` implements three key training optimizations:

### 1. RMSNorm (Root Mean Square Normalization)
RMSNorm replaces standard `LayerNorm`. It scales activations purely by their root mean square instead of calculating both the mean and variance:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{Mean}(x^2) + \epsilon}} \times \gamma$$
Because it drops the mean subtraction step and learnable bias offsets ($\beta$), it reduces computational overhead by ~10% with zero loss in training accuracy.

### 2. Cosine Learning Rate Decay with Warmup
Rather than keeping the learning rate constant, training follows a dynamic cosine schedule:
*   **Linear Warmup:** For the first 5% of training steps, the learning rate ramps up linearly from 0 to `args.lr`. This stabilizes weights and prevents early gradient explosions.
*   **Cosine Decay:** For the remaining 95% of steps, the learning rate decays along a cosine curve down to 10% of its peak value, allowing the model to smoothly converge to a stable minima.

### 3. Weight Decay Exclusion
To prevent over-regularization of word coordinates and normalization gains:
*   L2 weight decay is applied only to multi-dimensional projection weights (e.g., `nn.Linear` weight matrices).
*   Weight decay is explicitly **excluded** for 1D gains/weights (`RMSNorm.weight`), biases, and vocabulary token embeddings (`wte.weight`).

---

## ⚡ PyTorch 2.x & GPU Hardware Optimizations

To train the 49.8M parameter `v4-Large` model efficiently on consumer GPU hardware (like the **NVIDIA GeForce RTX 5060 Ti**), `v4` integrates several advanced compute and memory optimizations:

### 1. Automatic Mixed Precision (AMP) with `bfloat16`
Using `torch.amp.autocast`, the training script automatically performs the forward/backward activation math in 16-bit half-precision (`bfloat16` uses 2 bytes instead of 4).
*   **Memory Savings:** Halves the activation memory footprint, lowering peak VRAM requirements from over 15 GiB to **~7.8 GiB** at batch size 16.
*   **Hardware Acceleration:** Accesses dedicated **Tensor Cores** on NVIDIA GPUs, yielding a **1.5x speedup** in raw compute.
*   **ONNX Stability:** Autocast only scales activations dynamically. The model weights themselves remain in `float32`, ensuring the exported ONNX model is fully compatible with standard `float32` Java runtime environments.

### 2. JIT Graph Compilation (`torch.compile`)
PyTorch 2.x JIT model compilation is enabled via `torch.compile(model)`. It intercepts the PyTorch model graph and compiles it into optimized CUDA kernels (using the OpenAI Triton compiler) before training:
*   **Kernel Fusion:** Combines adjacent operations (like SwiGLU projections, element-wise multiplications, and RMSNorm scaling) into single GPU execution calls.
*   **Eliminates Launch Overhead:** At smaller batch sizes (like 8 or 16), PyTorch training is heavily bottlenecked by CPU-to-GPU kernel launch latency. Kernel fusion reduces these calls, resulting in a **2.3x speedup** in training throughput.

### 3. Pinned Memory & Workers Dataloading
The PyTorch `DataLoader` is optimized with `pin_memory=True` and `num_workers=2`:
*   **Page-Locked Memory:** Allocates CPU tensors in pinned memory, allowing high-speed, direct memory access (DMA) transfers to GPU VRAM.
*   **Asynchronous Prefetching:** Utilizes background CPU threads to load and prepare the next batches, ensuring the GPU never sits idle waiting for data.

### 4. Dataset Stride Optimization (Stride 512)
Instead of extracting training sequences with a sliding stride of `10` (which resulted in 99.0% identical overlapping sequences), `v4` uses a stride of **`512`** (50% overlap of the 1,024 context window):
*   **Eliminates Redundancy:** Reduces the total batch count per epoch by **51.2x** while still exposing the model to 100% of the text corpus twice per epoch.
*   **Performance Impact:** Accelerates epoch times from ~3.6 hours down to **~76 seconds** (under bfloat16 + compile), reducing the total 40-epoch training time from 6 days to **under an hour** while improving model generalization.

---

## 🏗️ Project Structure & Component Mappings

*   **[tokenizer.py](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv4/tokenizer.py)**: Python implementation of BBPE Tokenizer. Can save/load in the exact binary format used by Java's `BPETokenizer`.
*   **[train.py](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv4/train.py)**: The PyTorch trainer. Sets up BBPE tokenization, loads and cleans text files, trains the multi-layer Causal Transformer with cross-entropy loss and AdamW, and exports the final model to `model.onnx`.
*   **[OnnxLanguageModel.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv4/src/main/java/com/learnai/words/nn/OnnxLanguageModel.java)**: Loads `model.onnx` using the Java ONNX Runtime library. Feeds input tokens as a 2D tensor and retrieves output logits.
*   **[TextGenerator.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv4/src/main/java/com/learnai/words/nn/TextGenerator.java)**: Auto-regressive text generator executing temperature softmax and Top-K candidate sampling on the ONNX model output logits.
*   **[PromptCLI.java](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv4/src/main/java/com/learnai/words/cli/PromptCLI.java)**: Interactive terminal prompt client.

---

## ⚡ Setup & Execution Guide

### 1. Python Environment Setup
The python trainer requires `torch`, `onnx`, `onnxscript`, `numpy`, and `regex`. Build a local virtual environment:
```bash
python3 -m venv venv
venv/bin/pip install torch onnx numpy regex onnxscript
```

### 2. Model Training & ONNX Export
Run the retraining script to wipe out old artifacts and train on your cleaned corpus:
```bash
./retrain_model.sh
```
*(Configurable parameters like batch size, block size, epochs, and dims can be adjusted inside the [train_model.sh](file:///home/markvasey/Dropbox/GitHub/mystuff/LearnAI-Wordsv4/train_model.sh) script).*

### 3. Interactive Prompt CLI (Java)
Run the Java CLI to load `model.onnx` and interactively generate text:
```bash
./prompt_model.sh
```

### 4. Running Unit Tests
Compile and verify the test suite (which validates BBPE tokenization and checks ONNX model inference if `model.onnx` is present):
```bash
./mvnw test
```

---

## 🧮 Understanding Training Loss & Perplexity

During training, the console logs the **Cross-Entropy Loss** for the batch. Here is what this loss represents and how it is calculated:

### 1. Intuition: What Loss Represents
Loss measures **how wrong or surprised** the model is when trying to predict the next word in a sequence.
*   **High Loss (e.g., ~8.4 at startup)**: The model is highly confused. It assigns small, near-equal probabilities to all words in the vocabulary.
*   **Low Loss (e.g., < 4.0)**: The model is confident and predicting the correct words with high probability.

We measure the size of the model's "uncertainty guessing pool" using **Perplexity ($PPL$)**:
$$\text{Perplexity} = e^{\text{Loss}}$$
*   **At startup (Loss = 8.4)**: $e^{8.4} \approx 4,447$. The model is as confused as if it had to choose uniformly from a pool of ~4,447 words (its entire vocabulary).
*   **During training (Loss = 4.9)**: $e^{4.9} \approx 134$. The model has narrowed down its search space to a pool of ~134 likely candidate words.

### 2. Mathematics: How Loss is Calculated
Language models utilize **Cross-Entropy Loss** to calculate prediction error. 

At each sequence position, the model outputs raw scores (logits) for each vocabulary item. We apply the **Softmax** function to convert these logits into a probability distribution $P$ where $\sum P = 1.0$.

If the actual target next token in the text is index $y$, the loss for that specific token prediction is:
$$\mathcal{L} = -\ln(P(y))$$

*   **Perfect Prediction**: The model assigns a probability of $1.0$ to the correct token:
    $$\mathcal{L} = -\ln(1.0) = 0.0$$
*   **Weak Prediction**: The model only assigns a probability of $0.01$ ($1\%$) to the correct token:
    $$\mathcal{L} = -\ln(0.01) \approx 4.6$$
*   **Random Guessing**: At startup, weights are random, so the model assigns an equal probability to each of the $4,096$ vocab classes ($P(y) = 1/4096$):
    $$\mathcal{L} = -\ln(1/4096) \approx 8.3$$

The reported **Train Loss** is the average of these individual token cross-entropies across all tokens in the batch (batch size $\times$ sequence length).

---

## 📈 Case Study: v2 vs. v4 Performance & Learning

A direct comparison of training the **Fictional Literature** dataset (2.36M cleaned tokens) on an NVIDIA GPU using the old v2 custom CUDA framework versus the new v4 PyTorch-to-ONNX pipeline:

### 1. Model Configuration & Performance

| Metric | LearnAI-Wordsv2 (Custom CUDA) | LearnAI-Wordsv4 (PyTorch + ONNX) | Improvement / Shift |
| :--- | :---: | :---: | :--- |
| **Active Parameters** | 3,220,992 (3.22M) | **49,820,160 (49.8M)** | ~15x capacity increase |
| **Attention Mechanism** | Single-head Attention | **8-head Grouped-Query Attention (GQA)** | 4x faster KV Cache during generation |
| **Positional Encoding** | Absolute Position Vectors | **Rotary Position Embeddings (RoPE)** | Better relative distance, infinite context scale |
| **Normalization** | LayerNorm | **RMSNorm** | ~10% faster computation time |
| **Feed-Forward Blocks** | Single Linear Layer | **SwiGLU Non-linear MLP** ($512 \to 2048 \to 512$) | Higher expressive capacity |
| **Token Throughput** | ~23,040 tokens/s | **~112,640 tokens/s** | **4.9x more tokens processed per second** |
| **Sequences Throughput** | ~360 seq/s (size 64) | **~110 seq/s (size 1024)** | Handles 16x larger context window |
| **Epoch Duration** | ~248 seconds | **~76 seconds** | 3.2x faster epoch execution |
| **Best Val Loss (Epoch 1)** | — (uncalculated) | **`5.3250`** | Rapid syntactic alignment |
| **Best Val Loss (Epoch 12)** | — | **`0.3681`** | Optimal convergence before early stopping |

### 2. Dialogue & Character Learning Insights (by Epoch 1)

An output sample generated by `LearnAI-Wordsv4` at the very first epoch illustrates the immediate architectural gains of BBPE and text cleansing:

```text
Sample (Epoch 1): [The , and
and, I’ll be to the time. He was a little of
to the same I would be to be a great of the other.”

“What, I’ve the time; and I said, I’]
```

*   **Zero `<UNK>` Pollution**: Unlike `v3` where dialogue punctuation or contractions (e.g. `don<UNK>t`) were replaced by raw `<UNK>` tokens due to character-level BPE limits, `v4` natively outputs curly double quotes (`“` / `”`) and curly apostrophes (`I’ll`, `I’ve`) from Epoch 1.
*   **Gutenberg Metadata Elimination**: The regex-based text cleansing successfully keeps the training sequences free of Gutenberg index headers, transcriber labels, or random page numbers.
*   **Dialogue Conventions**: Even at Epoch 1, the model is already learning to nest dialogue segments on new lines and matches opening and closing double quotation marks.

### 3. Highly Converged Narrative & Conversational Spacing (by Epochs 12–14)

By Epoch 12, the validation loss reached its minimum at **`0.3681`** (a perplexity of $e^{0.3681} \approx 1.44$). Subsequent epochs (13–15) saw the training loss drop further (`0.27` $\to$ `0.20`) while the validation loss drifted up, indicating that the model began to overfit on the 2.36M token corpus. Early stopping triggered at Epoch 15 (patience 3/3), and the trainer automatically restored the best Epoch 12 weights for the final `model.onnx` export.

Representative samples from these final optimized epochs show mature literary style:

```text
Sample (Epoch 12): [The no-fit,” said Alice, as they approached her for the seat and
turned to touch them together.

“This is the same table,” said Miss Pross, “and unless a little white
hair got off its tail, you]
```

```text
Sample (Epoch 14): [The I. They spoke.”

“Capital! Don’t speak about those words.”

“Tell us what it is.”

“My dear fellow, I am not quite sure that I have heard them say. But it is]
```

*   **Multi-Speaker Conversational Layout**: The model has mastered narrative breaks and conversational flow, separating different speakers into individual lines with double quotation marks and proper capitalization.
*   **Zero Character Corruption**: Contractions (`Don’t`, `I’ll`) and dialogue quotes are perfectly rendered in UTF-8 bytes without `<UNK>` placeholders.
*   **Novel-Specific Name and Context Association**: The model draws together distinct characters and environments from its corpus, mixing references to **Alice** (*Alice in Wonderland*), **Miss Pross** (*A Tale of Two Cities*), and **Darcy** (*Pride and Prejudice* in Epoch 9) within a grammatically consistent structure.

---

## 🧠 Deep Dive: Grouped-Query Attention (GQA) & RoPE

To understand **GQA** and **RoPE**, think of them as upgrades to the attention "spotlight" and coordinate systems:

### 1. Grouped-Query Attention (GQA)
In standard attention, a model uses multiple attention heads to look at different concepts simultaneously. 
*   **MHA (Multi-Head):** For 8 Query heads ($Q$), you have 8 Key heads ($K$) and 8 Value heads ($V$). During token generation, you must store all previous keys/values in memory (the **KV Cache**). For multiple users or long sequences, this cache consumes gigabytes of VRAM.
*   **GQA (Grouped-Query):** Groups Query heads to share Key/Value heads. In our configuration, we group the 8 Query heads into 2 groups of 4. Each group shares a single Key and Value head (8 Q, 2 K, 2 V). 
*   **The Pro:** The KV Cache is reduced by **4x**, saving massive amounts of VRAM and speeding up memory bandwidth, while preserving almost the exact same language capacity and accuracy as MHA.

### 2. Rotary Position Embeddings (RoPE)
Instead of adding fixed position vectors to word embeddings, RoPE applies a rotation to the Query and Key vectors in 2D pairs.
*   The angle of rotation is proportional to the word's index.
*   **The Pro:** When the model calculates attention ($Q K^T$), the trigonometric properties of the rotated vectors cause the calculation to depend natively on the **difference** in positions (relative distance: $m - n$), rather than absolute coordinates. This lets the model understand grammar rules (like "subject precedes verb") regardless of where the clause appears in a 1,000-token window.

---

## 🧮 Math Deep Dive: Stacking Layers, Activations, and Non-Linearity

A common point of confusion is whether stacking more layers of weight matrices natively allows a neural network to model non-linear relationships.

### 1. The Matrix Collapse Rule (Why Matrix Stacking is Linear)
In linear algebra, multiplying matrices together always results in a matrix. If you stack three layers of weights ($W_1, W_2, W_3$) and multiply them by your input vector $X$:
$$Y = W_3 \cdot (W_2 \cdot (W_1 \cdot X))$$

You can group the matrices together into a single combined matrix:
$$W_{\text{combined}} = W_3 \cdot W_2 \cdot W_1$$
$$Y = W_{\text{combined}} \cdot X$$

Consequently, a deep network composed of pure matrix multiplications is mathematically identical to a single-layer linear model. It can only draw straight lines (linear separations) and cannot model curves, logical XOR gates, or complex syntactic hierarchies.

### 2. Bending Space with Activation Functions
To prevent the stacked matrices from collapsing, we introduce a **non-linear activation function** between the multiplications. In `v4`, we use the **SwiGLU** activation.

#### SwiGLU (Swish Gated Linear Unit) — The State-of-the-Art Choice
SwiGLU is a gated linear unit that uses the Swish activation function. It is defined as:
$$\text{SwiGLU}(x) = \left(\text{Swish}(x W_1) \otimes (x W_2)\right) W_3$$
Where $\text{Swish}(x) = x \cdot \sigma(\beta x)$ (also known as SiLU in PyTorch, with $\beta=1$), and $\otimes$ is element-wise multiplication.

*   **Behavior**:
    *   It uses a gating mechanism where one linear projection ($W_1$) scaled by the non-linear Swish function acts as a gate to control the flow of information from the second linear projection ($W_2$).
    *   This dynamic gating provides much higher expressive capability and smoother gradient flow compared to standard MLPs, leading to faster training convergence and lower validation perplexity per parameter count.

### 3. Hierarchical Feature Extraction
When we stack multiple layers of matrices separated by activations, the model builds a hierarchy of abstract understandings:
*   **Early Layers**: Map local characters to word shapes and simple punctuation tokens.
*   **Middle Layers**: Match syntactic relations, such as ensuring correct verb tenses or mapping dialogue punctuation conventions.
*   **Deep Layers**: Capture global, long-range semantic context (e.g. tracking who is speaking across a paragraph or maintaining the literary style of the corpus).

### 4. Quadratic Complexity in Self-Attention
In a Transformer model, a true **quadratic ($O(T^2)$) relationship** is introduced by the **Self-Attention mechanism**:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

When calculating the score matrix $Q K^T$, the model multiplies the representations of tokens with *each other* ($X W_Q \cdot W_K^T X^T$), rather than just multiplying tokens by static weight matrices. This pairwise multiplication means that every token in a sequence of length $T$ compares itself to every other token, scaling quadratically with the sequence length and enabling dynamic, contextual associations across the entire context window.

---

## 🧮 Model Parameter Breakdown & Embedding Semantics

A common question is how parameters are calculated in a Transformer model and whether the **token embedding layers** should count toward the parameter budget or be dismissed as "just a static lookup table of words."

### 1. Mathematical Breakdown of the 49.8M Model
For our active config (`vocab_size` = 8,192, `d_model` = 512, `block_size` = 1,024, `n_layer` = 12, `n_head` = 8, `n_kv_head` = 2), the exact parameter counts are:

| Component | Dimensions / Formula | Parameter Count |
| :--- | :--- | :---: |
| **Token Embeddings (`wte`)** | $\text{vocab\_size} \times d_{\text{model}} = 8,192 \times 512$ | $4,194,304$ (4.19M) |
| **Position Embeddings (`wpe`)** | *Removed in favor of Rotary Position Embeddings (RoPE)* | $0$ (0.00M) |
| **12x Blocks (Attention + SwiGLU)** | $12 \times (\text{RMSNorms} + \text{Self-Attention} + \text{SwiGLU})$ | $45,625,344$ (45.63M) |
| **Final RMSNorm (`ln_f`)** | Weight only = $512$ | $512$ (<0.01M) |
| **Language Model Head (`lm_head`)** | $\text{vocab\_size} \times d_{\text{model}} = 8,192 \times 512$ | $4,194,304$ (4.19M) |
| **Total (Without Weight Tying)** | Sum of all layers | **$54,014,464$ (54.0M)** |
| **Total (With Weight Tying)** | Shared `wte` and `lm_head` weights | **$49,820,160$ (49.8M)** |

*   **Excluding Embeddings:** If you only count the "core processing layers" (excluding token embeddings and the prediction head), the parameter count drops to **45,625,856** (~45.6M).

### 2. Do Token Embeddings Count as Parameters?
Yes, absolutely. They are not static mappings; they are trainable floating-point weights. During training, the gradients flow all the way back to the embedding matrix, updating the 4,194,304 variables inside `wte.weight` with every batch.

### 3. Do Embeddings Model Semantics or Just Words?
Token embeddings represent **lexical semantics** (static meaning and association), while the attention layers model **contextual semantics** (how meaning shifts based on surrounding words).

As training progresses, the model maps words into a 512-dimensional space where:
*   **Semantic Clustering:** Words that share semantic categories (e.g. names, verbs of action, types of places) develop coordinates that cluster close together (measured by high cosine similarity).
*   **Geometric Offsets:** Relational patterns (e.g. present vs. past tense, masculine vs. feminine pronouns) map to consistent vector offsets:
    $$\vec{v}_{\text{walked}} - \vec{v}_{\text{walk}} \approx \vec{v}_{\text{jumped}} - \vec{v}_{\text{jump}}$$
*   **Weight Tying Alignment:** By tying `self.wte.weight = self.lm_head.weight`, the model forces the input representation space and output prediction space to be perfectly aligned. A token cannot be predicted accurately if its input semantic representation has not converged.
