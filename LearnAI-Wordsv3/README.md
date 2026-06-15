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

## 📈 Case Study: v2 vs. v3 Performance & Learning

A direct comparison of training the **Fictional Literature** dataset (2.69M tokens) on an NVIDIA GPU using the old v2 custom CUDA framework versus the new v3 PyTorch-to-ONNX pipeline:

### 1. Model Configuration & Performance

| Metric | LearnAI-Wordsv2 (Custom CUDA) | LearnAI-Wordsv3 (PyTorch + ONNX) | Improvement / Shift |
| :--- | :---: | :---: | :--- |
| **Active Parameters** | 3,220,992 (3.22M) | **4,273,664 (4.27M)** | ~33% capacity increase |
| **Attention Mechanism** | Single-head Attention | **4-head Multi-Head Attention** | Better contextual mapping |
| **Feed-Forward Blocks** | Single Linear Layer | **GELU Non-linear MLP** ($256 \to 1024 \to 256$) | Higher expressive capacity |
| **Throughput (seq/s)** | ~360 seq/s | **~912 seq/s** | **2.5x faster** raw speed |
| **Dataset Volume** | Capped at 90,000 sequences | **Full 230,656 sequences** (uncapped) | Trains on $100\%$ of the corpus |
| **Epoch Duration** | ~248 seconds | **~267 seconds** | Same duration, but 2.5x more data |
| **Best Val Loss (Epoch 5)** | — (uncalculated) | **`3.4134`** | Stronger early convergence |
| **Best Val Loss (Epoch 11)** | — | **`2.8899`** | Deep semantic compression |
| **Best Val Loss (Epoch 14)** | — | **`2.7255`** | Advanced syntactic modeling |
| **Best Val Loss (Epoch 40)** | `4.3553` | **`< 2.65`** (estimated) | Beat v2's best score by Epoch 5 |

### 2. Dialogue & Character Learning Insights (by Epoch 5)

An output sample generated by `LearnAI-Wordsv3` at Epoch 5 illustrates the advanced grammatical structures learned by the model:

```text
Sample (Epoch 5): [The 50.<UNK>

<UNK>What is the matter?<UNK>

<UNK>I don<UNK>t know what I say is,<UNK> said Holmes, <UNK>I have not a
pale to do it.<UNK>

<UNK>I have a right,<UNK> answered]
```

*   **Punctuation Mapping via `<UNK>`**: Because the base tokenizer represents ASCII characters 0–255, any character $\geq 256$ (like Gutenberg's curly double-quotes `“` / `”` or curly apostrophes `’`) is mapped to the `<UNK>` token (ID 256). The model has successfully learned to predict `<UNK>` where punctuation contractions belong (e.g. `don<UNK>t`) and to wrap dialogue sections correctly (`<UNK>What is the matter?<UNK>`).
*   **Semantic & Character Association**: The model has learned specific nouns and character relationships from the corpus—such as associating dialogue blocks with the name **Holmes** (from the Sherlock Holmes collection).

### 3. Coherent Text Generation & Grammar (by Epoch 11)

By Epoch 11, the validation loss dropped to **`2.8899`** (reducing the model's word prediction uncertainty pool to only ~18 words). A generated text sample illustrates the growth in coherence:

```text
Sample (Epoch 11): [The verses were in the same room, and I saw a gentleman sitting in the
dark room, looking at me, and said, <UNK>I am not quite happy to allow to
myself that I am in the castle of the castle.<UNK>

]
```

*   **Dialogue Conventions**: The model has learned complex grammatical conventions, such as placing a comma and space right before direct speech (`...looking at me, and said, <UNK>I am...`).
*   **Quote Matching**: It successfully matches and closes double quotation marks (`<UNK>`) across multi-line splits.
*   **Vocabulary Variety**: It dynamically handles and generates vocabulary like `"verses"`, `"gentleman"`, `"castle"`, and `"allow to myself"`.

### 4. Advanced Phrase Modeling & Syntactic Clauses (by Epoch 14)

By Epoch 14, the validation loss dropped to **`2.7255`** (reducing the model's prediction uncertainty pool to ~15 words). The generated sample demonstrates the model's expanding ability to form complex structures:

```text
Sample (Epoch 14): [The ingratitude, the
fingers of his owner, was not so high, and his eyes, and his head was
turned with a small instrument<UNK>s nose, as if he were lying on the
f]
```

*   **Possessive Genitive Mapping via `<UNK>`**: At earlier epochs, `<UNK>` was primarily mapped to contractions (`don<UNK>t`) or dialogue quotes. By Epoch 14, the model successfully models genitive possessives (`instrument<UNK>s` representing `instrument’s`), showing high precision in punctuation context identification.
*   **Abstract & Sophisticated Vocabulary**: The model successfully handles and places complex abstract nouns (like `"ingratitude"`) alongside concrete descriptions (`"instrument"`, `"owner"`).
*   **Complex Sentence Structures**: The generation showcases sophisticated syntactic structures, including an appositive clause (`The ingratitude, the fingers of his owner...`), parallel subject groupings (`and his eyes, and his head...`), and a conditional adverbial clause (`as if he were lying...`).

### 5. Architectural Drivers of v3's Efficiency (Why it is 10x Smarter)

While the parameter count of `v3` is only **32.7% larger** than `v2` (4.27M vs 3.22M), `v3` converges in a fraction of the time to a much lower loss. The secret behind this efficiency lies in three core architectural features:

*   **Tied Embeddings & LM Head (Weight Tying)**: In `v2`, the input token embeddings and output Language Model Head were two separate matrices (each containing 1.05M parameters). In `v3`, these weights are shared (tied). Weight tying forces the model to map words to a semantic space aligned directly with word predictions, speeding up convergence and preventing representation mismatch. Without weight tying, `v3` would require 5.3M parameters.
*   **GELU MLP ($256 \to 1024 \to 256$) vs. Single Linear Layer**: The feed-forward path in `v2` was a simple linear projection with no non-linear activation function. Multiple linear operations collapse mathematically into a single linear mapping, severely limiting the model's feature-combining capabilities. `v3` introduces an actual MLP with a **GELU activation** and $4\times$ hidden state expansion, functioning as a non-linear logic gate to learn complex grammar rules.
*   **4-Head vs. Single-Head Attention**: While `v2`'s single attention head was forced to compromise on a single context spotlight per token, `v3`'s 4 heads divide the hidden representation into independent subspaces, enabling the model to track 4 distinct relationships (e.g. subject, object, verb tense, and punctuation context) simultaneously.

---

## 🧠 Deep Dive: Multi-Head Attention (4-Head)

To understand **4-Head Attention**, think of it as a sliding spotlight system:

### 1. Single-Head vs. Multi-Head Attention
In a Transformer, **Self-Attention** is the mechanism that allows a word to look at other words in a sentence to figure out its own context.

*   **Single-Head Attention (v2)**: Acted as a **single spotlight**. When processing a word, the model could only focus on one primary relationship at a time.
    *   *Example*: In *"The **bank** of the **river** was muddy"*, the spotlight on **"bank"** shines on **"river"** to determine it means a land bank. But if other relationships exist in the sentence, the single spotlight is forced to compromise.
*   **4-Head Attention (v3)**: Splits the model's hidden representation ($d_{model} = 256$) into **4 independent subspaces** ($256 / 4 = 64$ dimensions each). Each subspace gets its own attention calculator (a "head"), effectively giving the model **four independent spotlights** to shine simultaneously.

### 2. Spotlights in Action
Consider this sentence:
> **"The scientist examined the bacteria with a microscope."**

If we look at the word **"examined"**, the 4 heads can focus on completely different semantic aspects of the sentence at the same time:
*   **Head 1 (Subject Spotlight)**: Focuses on **"scientist"** (learning *who* did the action).
*   **Head 2 (Object Spotlight)**: Focuses on **"bacteria"** (learning *what* was acted upon).
*   **Head 3 (Instrument Spotlight)**: Focuses on **"microscope"** (learning *how* the action was done).
*   **Head 4 (Grammatical Spotlight)**: Focuses on verb tense agreements and punctuation context.

### 3. The Mathematics of 4-Head Attention
1.  **Project**: The input tokens of dimension $256$ are multiplied by weight matrices to create Queries ($Q$), Keys ($K$), and Values ($V$) vectors.
2.  **Split**: The vectors are split into $4$ chunks of $64$ dimensions each.
3.  **Attention**: Each head calculates attention scores independently on its own $64$-dimensional chunk:
    $$\text{Attention}(Q_i, K_i, V_i) = \text{Softmax}\left(\frac{Q_i K_i^T}{\sqrt{64}}\right) V_i$$
4.  **Concatenate**: The outputs of all 4 heads (each size 64) are glued back together:
    $$\text{Output} = [\text{Head}_1, \text{Head}_2, \text{Head}_3, \text{Head}_4] \rightarrow \text{Dimension } 256$$
5.  **Project**: The combined vector is projected through a final linear layer to mix information across all heads.

This multi-dimensional attention is why `LearnAI-Wordsv3` converges faster and learns complex structures (like quotes and character dialogue) much more effectively than the single-headed model.

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
To prevent the stacked matrices from collapsing, we introduce a **non-linear activation function** (like `GELU` or `ReLU`) between the multiplications:
$$Y = W_2 \cdot \text{GELU}(W_1 \cdot X)$$

Because these functions are non-linear, they "bend" the values. This bends the underlying mathematical space, preventing the layers from collapsing. According to the **Universal Approximation Theorem**, stacking these bent spaces allows a neural network to approximate *any* mathematical curve or decision boundary.

Two common activation functions include:

#### A. ReLU (Rectified Linear Unit) — The Benchmark
The simplest non-linear threshold gate is ReLU:
$$\text{ReLU}(x) = \max(0, x)$$

*   **Positive Input ($x > 0$)**: Passes the value unchanged (linear slope of 1).
*   **Negative Input ($x \le 0$)**: Outputs exactly 0 (flat line).

The sharp "bend" (or elbow) at $x = 0$ is the non-linearity. It acts like a switch: *"If a feature is negative, turn it off; if positive, let it pass."*

#### B. GELU (Gaussian Error Linear Unit) — The Modern Choice (v3)
GELU is a smoother, probabilistic gate that weights inputs by their likelihood of active utility. It is defined as:
$$\text{GELU}(x) = x \cdot \Phi(x)$$

Where $\Phi(x)$ is the cumulative distribution function (CDF) of the standard normal (Gaussian) distribution:
$$\Phi(x) = P(X \leq x), \quad \text{where } X \sim \mathcal{N}(0, 1)$$

*   **Behavior**:
    *   For highly positive inputs ($x \gg 0$), $\Phi(x) \to 1.0$, so $\text{GELU}(x) \approx x$ (linear pass-through).
    *   For highly negative inputs ($x \ll 0$), $\Phi(x) \to 0.0$, so $\text{GELU}(x) \approx 0$ (zero gate).
    *   Near zero, it forms a smooth curve, dipping slightly below zero (minimum of $\approx -0.17$ at $x \approx -0.76$). This smoothness prevents "dead neurons" (where a node outputs exactly zero and stops learning).
*   **Code Approximation Formula**: Calculating the exact Gaussian CDF is computationally expensive, so models approximate it in code as:
    $$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$$

### 3. Hierarchical Feature Extraction
When we stack multiple layers of matrices separated by activations, the model builds a hierarchy of abstract understandings:
*   **Early Layers**: Map local characters to word shapes and simple punctuation tokens.
*   **Middle Layers**: Match syntactic relations, such as ensuring correct verb tenses or mapping dialogue punctuation conventions.
*   **Deep Layers**: Capture global, long-range semantic context (e.g. tracking who is speaking across a paragraph or maintaining the literary style of the corpus).

### 4. Quadratic Complexity in Self-Attention
In a Transformer model, a true **quadratic ($O(T^2)$) relationship** is introduced by the **Self-Attention mechanism**:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

When calculating the score matrix $Q K^T$, the model multiplies the representations of tokens with *each other* ($X W_Q \cdot W_K^T X^T$), rather than just multiplying tokens by static weight matrices. This pairwise multiplication means that every token in a sequence of length $T$ compares itself to every other token, scaling quadratically with the sequence length and enabling dynamic, contextual associations across the entire context window.



