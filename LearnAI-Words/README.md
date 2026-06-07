# LearnAI-Words: Pure Java LLM from Scratch

`LearnAI-Words` is a character-level Large Language Model (LLM) implemented entirely in **Java 23**, without any external machine learning libraries. It demonstrates the underlying mathematics of the Transformer architecture through a clean, readable, and highly parallelized implementation.

---

## 🚀 Performance Breakthrough
Through surgical mathematical optimization and multi-threaded engineering, this project achieves production-grade speeds on standard CPU hardware:
- **10,000x Speedup**: Core math refactored from $O(N \cdot D^2)$ to $O(N \cdot D)$, reducing step times from minutes to **~2ms**.
- **Extreme Throughput**: Currently processing **~460 sequences per second** on a 16-core CPU.
- **14-Core Parallelism**: Leverages a custom `ForkJoinPool` to utilize 14 threads with non-blocking gradient calculations and synchronized Adam updates.
- **Allocation-Free Math**: Implemented **In-Place** matrix arithmetic and **Transpose-Free** multiplication to minimize Garbage Collection overhead.

---

## 📊 Model Specifications & Dimensions

The model is a robust, mathematically complete Transformer. Below are the structural details for the current configuration (`d_model=128`, `block_size=32`, `vocab_size=105`).

### Layer Stack (12 Layers Total)
1.  **Input Embedding** ($105 \times 128$)
2.  **Positional Encoding** (Fixed Sine/Cosine)
3.  **Transformer Block 1**:
    *   Layer Norm 1
    *   Causal Self-Attention ($128 \times 128 \times 3$)
    *   Residual Connection
    *   Layer Norm 2
    *   Dense Feed-Forward ($128 \times 128$)
    *   Residual Connection
4.  **Transformer Block 2**:
    *   Layer Norm 3
    *   Causal Self-Attention ($128 \times 128 \times 3$)
    *   Residual Connection
    *   Layer Norm 4
    *   Dense Feed-Forward ($128 \times 128$)
    *   Residual Connection
5.  **Final Layer Norm**
6.  **Language Head** ($128 \times 105$)

### Total Parameter Count: **159,593**
*   **Embeddings**: 13,440
*   **Attention Weights**: 98,304
*   **Dense/FFN Layers**: 46,569 (includes Head)
*   **Layer Norms**: 1,280
*   *Note: This excludes Adam optimizer moments ($m, v$), which double the weight memory during training.*

---

## 🏗️ Architecture & Design

### System Overview
This diagram shows how the raw text flows through the system to become a trained model and eventually generated text.

```mermaid
graph TD
    Data[Training Data: *.txt] --> Dataset[TextDataset]
    Dataset --> Tokenizer[CharacterTokenizer]
    
    subgraph "Core Model"
        Tokenizer --> LM[LanguageModel]
        LM --> Block[Transformer Block x2]
        Block --> Head[Linear Head + Softmax]
    end
    
    subgraph "Optimization Engine"
        Head --> Backprop[Backpropagation Engine]
        Backprop --> Adam[Adam Optimizer]
        Adam --> Matrix[Matrix Math Engine]
    end
    
    LM --> Persistence[model.bin]
    Persistence --> Generator[TextGenerator]
```

### The Transformer Layer Stack
The internal structure of the model, showing the flow of data through the attention and normalization layers.

```mermaid
graph TD
    Input[Input Tokens] --> Emb[Embedding Layer]
    Emb --> Pos[Positional Encoding]
    
    subgraph "Transformer Block"
        LN1[Layer Norm] --> Attn[Causal Self-Attention]
        Attn --> Res1[Residual Add]
        Res1 --> LN2[Layer Norm]
        LN2 --> FF[Feed-Forward/Dense]
        FF --> Res2[Residual Add]
    end
    
    Pos --> LN1
    Res2 --> Head[Output Head]
    Head --> Softmax[Softmax Layer]
    Softmax --> Output[Next Character Prediction]
```

---

## 📉 Training Sequence (Logic Flow)
How the model processes a single training step across multiple threads.

```mermaid
sequenceDiagram
    participant W as Worker Thread
    participant M as Language Model
    participant Mat as Matrix Engine
    participant O as Adam Optimizer

    W->>M: train(sequence, target)
    M->>Mat: forward(X)
    Mat-->>M: Probabilities
    M->>M: Calculate Loss
    M->>Mat: backward(outputGradient)
    Note over Mat: O(N) Optimized Calculus
    Mat-->>M: WeightsGradients
    M->>O: updateWeights(G)
    Note over O: Synchronized Momentum Update
    O-->>W: Step Complete
```

---

## 🧠 The Learning Cycle: What happens during an Epoch?

During each of the 50 epochs, the model "reads" through 50,000 sequences of classic literature to master the statistical patterns of the English language.

### 1. The Process (The Mechanics)
For every sequence processed across the 14 parallel threads:
*   **The Guess (Forward Pass):** The model takes a window of 32 characters and predicts the 33rd.
*   **The Correction (Backward Pass):** It calculates the error (Loss) between its guess and the actual character from the text. Using the **Chain Rule**, it traces this error back through all 12 layers to identify which weights were responsible.
*   **The Update (Optimization):** The **Adam Optimizer** nudges the 159,593 parameters in a direction that minimizes future errors.

### 2. The Purpose (The "Why")
*   **Pattern Refinement:** Early epochs teach the model basic structures (e.g., "words are separated by spaces"). Later epochs teach complex relationships (e.g., "The name 'Sherlock' is usually followed by 'Holmes'").
*   **Convergence:** Each epoch is a step down the "Loss Mountain" toward the most accurate possible version of the weights.
*   **Generalization:** By seeing characters in varying contexts across multiple books, the model learns the *rules* of English rather than just memorizing specific lines.

### 3. The Outcome (The Results)
*   **Intelligent Weights (`model.bin`):** A binary snapshot of the model's "brain." It transforms from random noise into a structured map of human language.
*   **Generative Power:** As training progresses, the end-of-epoch samples evolve from gibberish into structured words, sentences, and eventually, coherent thoughts.

---

## 🎓 Philosophical Goal: The "Smart Student" vs. "The Perfect Library"

A critical design decision in `LearnAI-Words` is the balance between **Memorization** and **Generalization**.

### 1. The Perfect Library (Memorization)
*   **The Goal:** Encode every character of the training set losslessly.
*   **The Math:** Requires ~2 parameters per training character. For our 1.2M character dataset, this would require a **~2.5M parameter model**.
*   **The Risk:** The model becomes a "lookup table." It can quote books verbatim but cannot form new thoughts or understand the rules of English.

### 2. The Smart Student (Generalization)
*   **The Goal:** Master English grammar, structure, and semantic meaning.
*   **The Strategy:** Use a smaller parameter count (160k - 500k) to force the model to learn **rules** rather than **data**. This is "Distributional Semantics"—deriving meaning by analyzing the relationships between words.
*   **The Current Bottleneck:** Our current `block_size` of 32 limits the model to a 5-word "memory." It can learn to spell, but it cannot yet grasp the meaning of a full sentence.

---

## 🗺️ Roadmap: The Path to Semantic Meaning

We are currently in **Phase 1**. Once the current 50-epoch run is complete, we will pivot to Phase 2 to transition from a "Spelling Bee Champion" to a "Smart Student."

| Feature | Phase 1 (Current) | Phase 2 (Target) | Why? |
| :--- | :--- | :--- | :--- |
| **`block_size`** | 32 | **128** | To "see" full sentences and understand context. |
| **`d_model`** | 128 | **192** | To provide more associative memory for concepts. |
| **Blocks** | 2 | **3** | To add a deeper layer of "reasoning" for long sequences. |
| **Compute Cost** | ~100s / Epoch | ~7-8 hrs / Epoch | Attention math scales quadratically ($N^2$) with context. |

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

---
*Created as part of the LearnAI series - Exploring Artificial Intelligence through fundamental engineering.*
