# LearnAI-Words: Pure Java LLM from Scratch

`LearnAI-Words` is a character-level Large Language Model (LLM) implemented entirely in **Java 23**, without any external machine learning libraries. It demonstrates the underlying mathematics of the Transformer architecture through a clean, readable, and highly parallelized implementation.

---

## 🚀 Performance Breakthrough
This project has been engineered for extreme efficiency on multi-core CPUs.
- **10,000x Speedup**: Optimized the mathematical bottlenecks in Layer Normalization and backpropagation, reducing step times from minutes to **~20ms**.
- **14-Core Parallelism**: Leverages a custom `ForkJoinPool` and Java's `parallelStream` to utilize 14 CPU cores simultaneously while maintaining system responsiveness.
- **O(N) Optimization**: Refactored core backpropagation logic from $O(N \cdot D^2)$ to $O(N \cdot D)$, enabling the training of deep models on consumer hardware without GPU acceleration.

---

## 🏗️ Architecture: The Decoder-Only Transformer
The model follows a GPT-style architecture designed for next-token prediction:

1.  **Embedding Layer**: Maps characters to a dense 64-dimensional vector space.
2.  **Positional Encoding**: Uses sine/cosine functions to inject sequence order into the model.
3.  **Transformer Blocks (x2)**:
    *   **Causal Self-Attention**: Allows tokens to communicate with previous tokens while masking the future.
    *   **Layer Normalization**: Stabilizes training by standardizing activations.
    *   **Residual Connections**: Prevents vanishing gradients by adding $x + Layer(x)$.
    *   **Feed-Forward (Dense) Layers**: Increases model capacity and non-linearity.
4.  **Softmax Head**: Outputs a probability distribution over the character vocabulary.

---

## 📉 Mathematics & Optimization

### Custom Math Engine (`Matrix.java`)
All matrix calculus is implemented from scratch. This includes optimized matrix multiplication ($i, k, j$ loop ordering), broadcasting, transpositions, and element-wise operations designed for cache locality.

### Adam Optimizer
The model uses **Adam (Adaptive Moment Estimation)** rather than standard SGD. It tracks:
- **First Moment ($m$)**: Mean of gradients.
- **Second Moment ($v$)**: Uncentered variance of gradients.
This allows for per-parameter learning rate adjustment, leading to much faster convergence on complex text patterns.

### Backpropagation
Every layer implements a mathematically rigorous `backward()` pass. The `CausalSelfAttentionLayer` calculates gradients through the softmax-attention scores and Query/Key/Value projections using the multivariate chain rule.

---

## 📚 Training & Observability

### Multi-Threaded Pipeline
The training loop is built on a non-blocking parallel architecture. Worker threads calculate gradients for individual sequences independently, with **synchronized Adam updates** ensuring thread-safe weight adjustments without sacrificing throughput.

### Observability Stack
- **Logging**: Integrated with **SLF4J and Logback**. Real-time monitoring is available in `training.log`.
- **Metrics**: Every 1,000 sequences, the model logs **Avg Step Time (ms)** to track performance.
- **Resilient Checkpointing**: The model automatically saves its state to `model.bin` every **10,000 sequences** and at the end of every epoch.

---

## 🛠️ How to Run

### Requirements
- **JDK 23**
- **Maven** (included via `./mvnw`)

### Start Training
```bash
./mvnw exec:java -Dexec.mainClass="com.learnai.words.cli.WordsCLI"
```

### Monitor Progress
```bash
tail -f training.log
```

### Generate Text
The program loads `model.bin` on startup if it exists, allowing you to resume training or generate text from a saved state. At the end of every epoch, the model outputs a 100-character sample to demonstrate its current learning progress.

---
*Created as part of the LearnAI series - Exploring Artificial Intelligence through fundamental engineering.*
