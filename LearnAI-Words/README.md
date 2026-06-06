# LearnAI-Words: Pure Java LLM from Scratch

`LearnAI-Words` is a demonstration of a character-level Large Language Model (LLM) implemented entirely in **Java 23**, without any external machine learning libraries (like PyTorch, TensorFlow, or DeepLearning4J). It is designed to show the underlying mathematics of the Transformer architecture through a clean, readable, and highly parallelized implementation.

## 🚀 Key Features
- **Zero Dependencies:** Built using only standard Java 23 and Maven for builds.
- **Custom Math Engine:** All matrix operations, including multiplication, transposition, and backpropagation, are implemented from scratch in `Matrix.java`.
- **Advanced Architecture:** A decoder-only Transformer featuring Causal Self-Attention, Layer Normalization, and Residual Connections.
- **Adam Optimizer:** Uses the production-grade Adam optimizer for stable and fast convergence.
- **CPU Parallelism:** Leverages Java's `parallelStream` and custom `ForkJoinPool` to distribute training across 14 CPU cores.

## 🏗️ Architecture: The Transformer Decoder
The model follows a GPT-style architecture (Decoder-only Transformer):

1.  **Embedding Layer:** Maps characters to a high-dimensional vector space ($d_{model} = 64$).
2.  **Positional Encoding:** Adds spatial information to character vectors using sine and cosine functions.
3.  **Transformer Blocks (x2):**
    *   **Layer Normalization:** Standardizes inputs to stabilize training.
    *   **Causal Self-Attention:** Allows the model to look at previous characters while masking "future" ones (using a triangular mask).
    *   **Residual Connections:** Adds the input of a layer back to its output ($x + Layer(x)$) to prevent vanishing gradients.
    *   **Feed-Forward (Dense) Layer:** Increases the non-linearity of the model.
4.  **Softmax Head:** Outputs a probability distribution over the entire vocabulary (characters) to predict the next token.

## 📉 Mathematics & Optimization
### Matrix Calculus
Every layer implements a `forward` and `backward` pass. The backward pass calculates gradients using the chain rule:
- **Attention Gradients:** Mathematically derived to propagate through the softmax scores and the query/key/value projections.
- **Dense Gradients:** Standard $dW = X^T \cdot dY$ implementation.

### Adam Optimizer
Rather than simple Gradient Descent, this project uses **Adam (Adaptive Moment Estimation)**. It maintains two moving averages for every weight:
1.  **m (First Moment):** The mean of the gradients.
2.  **v (Second Moment):** The uncentered variance of the gradients.
This allows the model to adjust the learning rate for each individual parameter, significantly speeding up training on text data.

## 📚 Training Process
### Data
The model trains on classic literature located in the `Training/` directory (e.g., Sherlock Holmes, Dorian Gray). It tokenizes text at the character level, making it robust to any language or style.

### Performance & Scaling
- **Parallelism:** The training process is "Ultra-Responsive." It distributes sequences across 14 threads.
- **Checkpoints:** The model automatically saves its state to `model.bin` every **10,000 sequences** and at the end of every epoch.
- **Logging:** Activity is tracked in `training.log`, providing updates every **1,000 sequences** with performance metrics (ms per step).

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
The program will automatically output a 100-character sample at the end of every epoch. The model loads `model.bin` on startup if it exists, allowing you to resume training or generate text from a pre-trained state.

---
*Created as part of the LearnAI series - Exploring Artificial Intelligence through fundamental engineering.*
