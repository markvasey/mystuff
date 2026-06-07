# LearnAI-Words: Pure Java LLM from Scratch

`LearnAI-Words` is a subword-level Large Language Model (LLM) implemented entirely in **Java 25**, without any external machine learning libraries. It is designed as a "glass box" for students and engineers to see exactly how modern AI works from the inside out.

---

## 🚀 Phase 2 Enhancements: The "Smart Student"
- **Subword BPE Tokenization**: A **1,000-token Byte Pair Encoding (BPE)** vocabulary allows the model to "think" in word parts. 
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

## 📂 Diagnostic Tools

### **1. Semantic Probe: Quantifying "Meaning"**
The `SemanticProbe` is a powerful tool to measure **Distributional Semantics**. It calculates the **Euclidean Distance** between the embedding vectors of two words.

*   **How it works:** In 192-dimensional space, the distance $d$ between two words $P$ and $Q$ is calculated as: $d(P, Q) = \sqrt{\sum_{i=1}^{192} (P_i - Q_i)^2}$.
*   **What it reveals:** If the distance between "king" and "queen" is lower than the distance between "king" and "carrot," the model has successfully learned that "king" and "queen" are semantically related (often appearing in similar contexts).
*   **Run it:**
    ```bash
    ./mvnw exec:exec -Dexec.arguments="--add-modules,jdk.incubator.vector,-classpath,%classpath,com.learnai.words.tokenizer.SemanticProbe"
    ```

### **2. Tokenizer Inspector**
See the subword fragments discovered by the BPE process.
```bash
./mvnw exec:exec -Dexec.arguments="--add-modules,jdk.incubator.vector,-classpath,%classpath,com.learnai.words.tokenizer.InspectTokenizer"
```

---

## 🏗️ The LanguageModel Engine

The `LanguageModel` class is the central nervous system. It orchestrates:
1.  **Prediction (Forward):** Passes data through Embedding -> Positional -> Transformer Blocks.
2.  **Learning (Backward):** Compares predictions to reality, calculates the **Error Gradient**, and sends it back through every layer to adjust the weights.

---

## 🛠️ How to Run

### 1. Train the Tokenizer (Required Once)
```bash
./mvnw exec:exec -Dexec.arguments="--add-modules,jdk.incubator.vector,-classpath,%classpath,com.learnai.words.tokenizer.BPETrainTool"
```

### 2. Start Model Training
```bash
./mvnw clean compile exec:exec
```

### 3. Monitor Progress
```bash
tail -f training.log
```

---

## 📚 Further Reading & Inspiration

To go deeper into the theory behind this implementation, explore these foundational resources:

- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)**: The original 2017 Google paper that introduced the Transformer architecture used in this project.
- **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)**: A brilliant visual guide by Jay Alammar that explains the math through diagrams.
- **[Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)**: A spiritual sibling to this project in Python/PyTorch, focused on making GPT simple and hackable.
- **[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)**: Michael Nielsen’s free online book—the gold standard for understanding Backpropagation and Gradient Descent.
- **[BPE (Byte Pair Encoding) Explained](https://huggingface.co/learn/nlp-course/chapter6/5)**: The HuggingFace guide to the subword tokenization strategy used in Phase 2.

---
*Created as part of the LearnAI series - Exploring Artificial Intelligence through fundamental engineering.*
