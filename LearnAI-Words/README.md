# LearnAI-Words: Pure Java LLM from Scratch

`LearnAI-Words` is a subword-level Large Language Model (LLM) implemented entirely in **Java 25**, without any external machine learning libraries. It is designed as a "glass box" for students and engineers to see exactly how modern AI works from the inside out.

---

## 🚀 Phase 2 Enhancements: The "Smart Student"
- **Subword BPE Tokenization**: A **1,000-token Byte Pair Encoding (BPE)** vocabulary allows the model to "think" in word parts. 
- **Robustness (UNK Token)**: Added a dedicated **`<UNK>` (Unknown) token (ID 256)** to handle exotic Unicode characters (like curly quotes) without crashing.
- **SIMD Optimized Matrix Engine**: Leverages the **Java Vector API** for a **2x performance boost** (100+ seq/s).
- **14-Core Parallelism**: Uses a custom `ForkJoinPool` for highly efficient training.

---

## 🧠 Intuition: How it Works

### **The "Word Spotlight" (Attention)**
Self-Attention is the "secret sauce" of LLMs. Think of it as a **dynamic spotlight**. In the sentence *"The **bank** of the **river** was muddy,"* the attention layer uses the word **"river"** to put a spotlight on **"bank."** This tells the model: *"In this specific sentence, 'bank' means land, not a financial building."*

### **The "Creativity Dial" (Temperature)**
When generating text, the model doesn't just pick the #1 answer. We use a "Creativity Dial" called **Temperature**:
- **Low Temperature (0.1):** The model is very focused and conservative. It will always pick the most likely word (e.g., "The quick brown fox").
- **High Temperature (1.2):** The model becomes "creative" (or "drunk"). It takes risks on less likely words, leading to more poetic or chaotic output.

### **The Semantic Space (Embeddings)**
Every word is assigned a location in a 192-dimensional "Semantic Space." Over time, the model learns to move related concepts closer together. Through training, the vector for "He" will naturally drift toward the vector for "She," and away from the vector for "Apple."

---

## 🎓 Learning Path: 3 Experiments to Run

Use this project as a laboratory to see AI principles in action:

1.  **The "Lobotomy" Experiment:** Train for only 1 epoch and generate text. You'll see "alphabet soup." Train for 50 epochs, and you'll see "word salad." Train for 150, and you'll see "sentences."
2.  **The "Context Stretch":** Find `BLOCK_SIZE` in `WordsCLI.java` and change it from 128 to 16. The model will suddenly "forget" the beginning of a sentence by the time it reaches the end. This proves why "Context Window" is so vital.
3.  **The "Brain Size" Test:** Double the `D_MODEL` from 192 to 384. Watch how training slows down but the "Avg Loss" drops much faster. This illustrates the trade-off between model capacity and compute cost.

---

## 🏗️ The LanguageModel Engine

The `LanguageModel` class is the central nervous system of the project. It doesn't just hold weights; it orchestrates the entire flow of information:

1.  **Orchestration (The Pipeline):** It passes data sequentially through the **Embedding**, **Positional Encoding**, and three **Transformer Blocks**.
2.  **Forward Pass (Prediction):** It takes a sequence of integers (tokens) and produces a probability distribution for the *very next* token.
3.  **Backward Pass (Learning):** This is where the "intelligence" happens. After comparing its prediction to the real target, it calculates the **Error Gradient**. It then sends this signal backward through every layer, telling each weight exactly how much it needs to change to be more accurate next time.
4.  **Global Gradient Clipping:** To prevent the model from "tripping" over large errors (which can cause `NaN` values), the `LanguageModel` normalizes the total gradient if it exceeds a threshold (1.0).

---

## 🎓 Worked Example: The Training Lifecycle

To understand how the "Smart Student" learns, let's trace a single piece of data through the system.

### **1. Tokens (The Alphabet of Concepts)**
Raw text is broken into BPE tokens.
*   **Text:** `"The quick brown fox"`
*   **Tokens:** `[45, 128, 512, 89]` (Integer IDs from `tokenizer.bin`)

### **2. Sequences (The Context Window)**
The model looks at a window of tokens called a **Sequence**. We use a `BLOCK_SIZE` of 128.
*   **Input Sequence:** `[45, 128, 512]` (The context: "The quick brown")
*   **Target Token:** `89` (The correct answer: "fox")

### **3. The Training Step (The "Aha!" Moment)**
1.  The model sees `[45, 128, 512]` and predicts probabilities for the next token.
2.  It might predict: `{"dog": 0.1, "fox": 0.05, "cat": 0.2}`.
3.  The **Loss Function** sees that the correct answer was `fox` (0.05) and calculates a high error because 5% is much lower than 100%.
4.  **Backpropagation** calculates the gradients, and **Adam** updates the weights to make the prediction for `fox` higher next time.

### **4. Epochs (The Full Curriculum)**
*   **One Training Step:** Learning from one sequence (context -> next word).
*   **One Epoch:** Learning from **every possible sequence** in your entire library (Sherlock Holmes, Dorian Gray, etc.).
*   **150 Epochs:** The model reads the entire library 150 times, refining its internal "grammar" and "logic" with every pass.

---

## 🎓 Technical Deep Dive: The Java Class Stack

### **1. Core Execution & Math**
*   **`WordsCLI.java`**: The main trainer. Orchestrates hyperparameters, multi-threaded training loops, and heartbeat monitoring.
*   **`Matrix.java`**: The high-performance math engine using the **Java Vector API (SIMD)**.

### **2. Tokenization & Data Handling**
*   **`BPETokenizer.java`**: Implements **Byte Pair Encoding**. Discovers subword patterns and handles `<UNK>` tokens.
*   **`BPETrainTool.java`**: Standalone tool used to "discover" the vocabulary and produce `tokenizer.bin`.
*   **`TextDataset.java`**: Slices raw books into clean training sequences and handles Gutenberg header removal.
*   **`InspectTokenizer.java`**: A diagnostic tool to visualize the subwords learned by the BPE process.

### **3. Neural Network Layers (`nn`)**
*   **`LanguageModel.java`**: The structural "brain" connecting all layers and managing the `train/predict` lifecycles.
*   **`CausalSelfAttentionLayer.java`**: The core context engine allowing tokens to attend to previous tokens.
*   **`EmbeddingLayer.java`**: Maps token IDs into 192-dimensional semantic vectors.
*   **`PositionalEncoding.java`**: Injects awareness of token order using sine/cosine waves.
*   **`DenseLayer.java`**: Standard fully-connected Feed-Forward layer.
*   **`LayerNorm.java`**: Stabilizes deep training by normalizing activations.
*   **`ResidualBlock.java`**: Implements "skip connections" to prevent vanishing gradients.
*   **`SoftmaxLayer.java`**: Converts raw scores (logits) into a probability distribution.
*   **`Adam.java`**: Adaptive optimizer tracking momentum and variance for every individual weight.
*   **`TextGenerator.java`**: Implements **Temperature** and **Top-K** sampling to generate creative text.

---

## 🧠 Mathematical Core: Softmax & Adam

### **Softmax: From Scores to Probabilities**
Softmax is the final "filter" that makes sense of the model's raw math. 
1. **Exponentiation ($e^x$):** Every raw score is raised to the power of $e$. This aggressively rewards confident guesses.
2. **Normalization:** Every value is divided by the total sum of all scores. This ensures the output is a valid probability distribution (summing to 100%).
3. **The Learning Signal:** The error (gradient) is remarkably simple: **Prediction - Target**.

### **Adam: The Adaptive Learner**
**Adam (Adaptive Moment Estimation)** maintains separate learning rates for every single parameter:
- **Momentum:** Remembers the *direction* of previous updates to carry speed through flat areas.
- **Scaling:** Tracks the *variance* of updates. If a weight is jumping wildly, Adam slows it down; if it's stable, Adam speeds it up.

---

## ⚡ Deep Dive: SIMD Vectorization (Java Vector API)
SIMD allows the CPU to process multiple numbers simultaneously in a single clock cycle.

**New SIMD Approach in `Matrix.java`:**
```java
for (; i < SPECIES.loopBound(arrayA.length); i += SPECIES.length()) {
    var va = DoubleVector.fromArray(SPECIES, arrayA, i);
    var vb = DoubleVector.fromArray(SPECIES, arrayB, i);
    va.add(vb).intoArray(arrayA, i);
}
```
This optimization doubled training throughput from ~45 seq/s to over 100 seq/s.

---

## 📂 Diagnostic Tools

### **1. Semantic Probe: Quantifying "Meaning"**
Calculates the **Euclidean Distance** between embedding vectors.
```bash
./mvnw exec:exec -Dexec.arguments="--add-modules,jdk.incubator.vector,-classpath,%classpath,com.learnai.words.tokenizer.SemanticProbe"
```

### **2. Tokenizer Inspector**
See the subword fragments discovered by the BPE process.
```bash
./mvnw exec:exec -Dexec.arguments="--add-modules,jdk.incubator.vector,-classpath,%classpath,com.learnai.words.tokenizer.InspectTokenizer"
```

---

## 🛠️ How to Run
1. **Train Tokenizer:** `./mvnw exec:exec -Dexec.arguments="--add-modules,jdk.incubator.vector,-classpath,%classpath,com.learnai.words.tokenizer.BPETrainTool"`
2. **Train Model:** `./mvnw clean compile exec:exec`
3. **Monitor:** `tail -f training.log`

---

## 📚 Further Reading
- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)**: Original Transformer paper.
- **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)**: Visual math guide.
- **[Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)**: Minimalist GPT in Python.
- **[BPE Explained](https://huggingface.co/learn/nlp-course/chapter6/5)**: Subword tokenization guide.

---
*Created as part of the LearnAI series - Exploring Artificial Intelligence through fundamental engineering.*
