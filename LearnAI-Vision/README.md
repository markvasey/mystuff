# LearnAI-Vision: Family Vision Transformer

A pure Java implementation of a **Vision Transformer (ViT)** built from scratch. This project demonstrates how deep learning architectures—specifically Self-Attention mechanisms—can be implemented using fundamental linear algebra without external libraries like TensorFlow or PyTorch.

## Project Overview

The primary goal of this project is to recognize family members from images. It employs a "patches-based" approach where images are broken down into a sequence of smaller sub-images (patches), which are then processed by a neural network that "attends" to different parts of the image simultaneously.

### Key Features
- **Custom Matrix Library**: A dedicated linear algebra implementation (`Matrix.java`) supporting dot products, transpositions, and element-wise operations.
- **Vision Transformer Architecture**: Implements patch embedding, self-attention, and global average pooling.
- **Multi-threaded Training**: Utilizes Java's `parallelStream()` to distribute image processing and gradient calculations across multiple CPU cores.
- **From-Scratch Backpropagation**: Full implementation of the backward pass for all layers, including the complex Self-Attention mechanism.

## Architecture

The model follows a sequential pipeline:
1. **Patch Embedding**: Converts an image (e.g., 32x32) into 16 patches (8x8 pixels each) and projects them into a 64-dimensional hidden space.
2. **Self-Attention Layer**: Allows patches to interact with each other, identifying which parts of the image (e.g., eyes, hair) are most relevant for classification.
3. **ReLU Activation**: Introduces non-linearity to the network.
4. **Global Average Pooling**: Summarizes the sequence of patch features into a single representative vector.
5. **Dense Classification Head**: Maps the summarized features to the final categories (family members).
6. **Softmax**: Converts raw output scores into probabilities.

## Getting Started

### Prerequisites
- **JDK 17 or higher** (The project uses modern Java features like `DoubleAdder` and `parallelStream`).

### Training the Model
The training images should be placed in the `Training/` directory. The model identifies labels based on the filename (e.g., `mark_01.jpg` is labeled as "mark").

To run the training:
1. Compile the source files.
2. Run `com.learnai.cli.VisionCLI`.

The CLI will:
- Load all `.jpg` and `.jpeg` images from the `Training/` folder.
- Initialize the model with Xavier/Glorot weights.
- Run 500 epochs of training using a multi-threaded loop.
- Output the average loss every 10 epochs.
- Perform a final validation check on the training set.

## Project Structure
- `com.learnai.math`: Fundamental linear algebra.
- `com.learnai.nn`: Neural network layers (`Dense`, `SelfAttention`, `Softmax`, etc.) and the core `NeuralNetwork` orchestrator.
- `com.learnai.vision`: Image loading, resizing, and patch extraction.
- `com.learnai.cli`: The main entry point for training and inference.
