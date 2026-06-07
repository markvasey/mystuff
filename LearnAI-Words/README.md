# LearnAI-Words: Pure Java LLM from Scratch

`LearnAI-Words` is a subword-level Large Language Model (LLM) implemented entirely in **Java 25**, without any external machine learning libraries. It demonstrates the underlying mathematics of the Transformer architecture through a clean, readable, and highly optimized implementation.

---

## 🚀 Phase 2 Enhancements: The "Smart Student"
We have recently transitioned from Phase 1 (Character-level) to **Phase 2**, introducing surgical optimizations and architectural scaling:
- **Subword BPE Tokenization**: A **1,000-token Byte Pair Encoding (BPE)** vocabulary allows the model to "think" in word parts (e.g., `["Sher", "lock"]`). 
- **Robustness (UNK Token)**: Added a dedicated **`<UNK>` (Unknown) token (ID 256)** to handle exotic Unicode characters (like curly quotes) without crashing, ensuring stable training across diverse texts.
- **SIMD Optimized Matrix Engine**: Leverages the **Java Vector API** (Incubator) to perform vectorized matrix multiplication. By processing multiple data points per CPU cycle (SIMD), we achieve a **2x performance boost**.
- **High Throughput**: Currently training at **100+ sequences per second** on a 13th Gen i7, reducing epoch time from 25 minutes to roughly **10 minutes**.
- **14-Core Parallelism**: Uses a custom `ForkJoinPool` with a real-time heartbeat monitor and per-epoch text generation samples.

---

## 📊 Model Specifications & Dimensions

The model is a deep Transformer optimized for semantic generalization over rote memorization.

### Layer Stack (16 Core Stages)
1.  **Input Embedding** ($1000 \times 192$)
2.  **Positional Encoding** (Fixed Sine/Cosine)
3.  **Transformer Blocks (x3)**:
    *   2x Layer Norms per block
    *   1x Causal Self-Attention ($192 \times 192 \times 3$)
    *   1x Dense Feed-Forward ($192 \times 192$)
    *   Residual connections and specialized initialization.
4.  **Final Layer Norm**
5.  **Language Head** ($192 \times 1000$)

### Total Parameter Count: **~636,000**
*   **Embeddings**: 192,000
*   **Attention Weights**: 331,776
*   **Dense/FFN Layers**: 110,592 (includes Head)
*   **Layer Norms**: 1,544

---

## 🏗️ Architecture & Design

### System Overview
This diagram shows how the raw text flows through the system to become a trained model and eventually generated text.

```mermaid
graph TD
    Data[Training Data: *.txt] --> BPE[BPE Discovery Tool]
    BPE --> Tokenizer[BPETokenizer]
    
    subgraph "Core Model (Phase 2)"
        Tokenizer --> LM[LanguageModel]
        LM --> Block[Transformer Block x3]
        Block --> Head[Linear Head + Softmax]
    end
    
    subgraph "Optimization Engine"
        Head --> Backprop[Backpropagation Engine]
        Backprop --> Adam[Adam Optimizer]
        Adam --> Matrix[SIMD Vectorized Engine]
    end
    
    LM --> Persistence[model.bin]
    Persistence --> Generator[TextGenerator]
```

---

## 🎓 Philosophical Goal: Generalization over Memorization

A critical design decision is the balance between **Memorization** and **Generalization**.

### The Smart Student Strategy
Instead of building a massive "Perfect Library" model that simply quotes books, we use a compact **636k parameter model**. This forces the model to learn the **rules** of English grammar and the **relationships** between subwords (Distributional Semantics) rather than just memorizing character sequences.

| Feature | Phase 1 (Legacy) | Phase 2 (Current) | Why? |
| :--- | :--- | :--- | :--- |
| **Tokenizer** | Character | **BPE (Subword)** | To capture semantic word parts. |
| **`block_size`** | 32 | **128** | To "see" full sentences and context. |
| **`d_model`** | 128 | **192** | More associative memory for concepts. |
| **Throughput** | ~50 seq/s | **~105 seq/s** | SIMD-accelerated Matrix math. |
| **Compute** | ~100s / Epoch | **~10 mins / Epoch** | Deeper context requires more math. |

---

## 🛠️ How to Run

### Requirements
- **JDK 25** (Required for the Vector API)
- **Maven** (included via `./mvnw`)

### Start Training
The project uses the `exec-maven-plugin` configured with the necessary JVM flags for the incubator Vector API.

```bash
./mvnw clean compile exec:exec
```

### Monitor Progress
```bash
tail -f training.log
```

---
*Created as part of the LearnAI series - Exploring Artificial Intelligence through fundamental engineering.*
