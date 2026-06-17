# Technical Architecture & Design Document: Multi-Modal Spatio-Temporal Transformer for Tonic-Clonic Seizure Detection

This document codifies the technical discussion and architectural blueprint for extending an existing computer vision pipeline to support deep-learning-based classification of generalized tonic-clonic seizures. By synthesizing the real-time processing capabilities of the Java-based video framework with the optimized deep learning training infrastructure of the language-modeling system, a non-causal Transformer encoder pipeline is established.

---

## 1. Engineering Feasibility & Architectural Paradigm

Transitioning from traditional rule-based threshold filters or image-heavy pixel models to a skeleton-based motion transformer is a highly credible architectural path. Restricting model inputs strictly to abstract spatial coordinates and continuous vector representations provides deep architectural advantages over raw pixel processing:

*   **Environmental and Lighting Invariance:** By extracting raw joint positions, the pipeline strips out background noise, furniture features, bedding color, and varying illumination profiles across different environments.
*   **Strict Privacy Guarantees:** In a sensitive medical telemetry context, storing and transmitting thin coordinate arrays completely mitigates the data security risks associated with storing or transferring raw RGB pixel streams of a patient in bed.
*   **Computational Efficiency:** Processing low-dimensional continuous vector matrices significantly scales down VRAM requirements and training parameters compared to heavy, resource-intensive 3D Convolutional Neural Networks (3D-CNNs).

---

## 2. Clinical Semiology & Feature Vector Mapping

Generalized tonic-clonic events carry unambiguous spatial and temporal signatures. To maximize detection reliability and minimize false alarms caused by everyday rhythmic movements (like walking or sitting), the input feature vector is structured as a 40-dimensional (or 39-dimensional if eye metrics are combined) continuous token combining skeletal tracking with high-fidelity facial metrics:

| Feature Subsystem | Dimensionality | Target Clinical Semiology |
| :--- | :---: | :--- |
| **Skeletal Joints (YOLOv8-Pose)** | 34 dimensions (17 joints × 2 coordinates) | Tracks bilateral extremities, rhythmic clonic shaking pacing ($2.0\text{ to }4.5\text{ Hz}$), and tonic rigid layout shifts. |
| **Head Rotation (PnP Solver via FaceMesh)** | 3 dimensions (Yaw, Pitch, Roll angles) | Isolates sudden, sustained forced lateral head deviation (versive movements) specific to tonic presentation. |
| **Eye Aspect Ratio (EAR)** | 2 dimensions (Left and Right ratios) | Calculates the ratio of vertical eyelid distance to horizontal length to track tonic wide-open eye locking. |
| **Jaw Aspect Ratio (JAR)** | 1 dimension (Vertical / horizontal mouth boundaries) | Identifies masseter contraction anomalies, distinguishing sudden jaw clenches or open-mouth spasms. |

---

## 3. Data Invariance via Resolution Normalization

Because training and validation data span multiple resolutions (e.g., 1080p, 720p) and varying aspect ratios, a uniform scale normalization step is required. This mathematical scaling prevents absolute pixel coordinate drift while completely preserving the structural geometry of the subject's posture:

```math
\text{MaxDim} = \max(\text{Width}, \text{Height})
```
```math
X_{\text{normalized}} = \frac{X_{\text{raw}}}{\text{MaxDim}}
```
```math
Y_{\text{normalized}} = \frac{Y_{\text{raw}}}{\text{MaxDim}}
```

Dividing both axes by the maximum dimension ensures all output points map uniformly to a $0.0$ to $1.0$ scale. This technique guarantees resolution-independence while keeping vertical indicators intact (allowing the model to explicitly separate standing/walking posture geometry from recumbent seizure configurations).

---

## 4. Four-Phase Production Architecture Blueprint

### Phase 1: Headless Batch Extraction (Java Client)
Extend the core Java system to scan the file-system dataset layout recursively (Training and Validation folder pairs). The batch processor bypasses UI components and uses the Foreign Function & Memory (FFM) Panama API structures to read files directly into off-heap memory. Frames execute sequentially through YOLOv8-Pose and the MediaPipe FaceMesh ONNX runtime. Coordinates are processed through uniform scale normalization and compiled into a single historical continuous matrix with a shape of `[TotalFrames, 39]`, dumped directly as structured JSON payloads next to source files.

### Phase 2: Sliding-Window Dataset Staging (Python DataLoader)
A custom PyTorch Dataset class maps the long extracted JSON coordinate matrices into discrete temporal evaluation slices. For a human-like response time of $3\text{ to }5\text{ seconds}$, setting a frame-skipping configuration to $10\text{ FPS}$ translates directly to a highly stable context window length of $T = 30\text{ to }50$ frames. Training segments use a short sliding stride and are fully shuffled to optimize gradient variations, while validation blocks remain sequence-pure to track exact generalization convergence curves.

### Phase 3: Non-Causal Spatio-Temporal Transformer Encoder
The continuous multi-modal feature matrix is mapped into a lower hidden dimension (e.g., 64 or 128) via a linear projection layer to minimize overfitting on the 1-hour video footprint. The layout features 2 to 3 dense Transformer encoder layers configured with:
*   **Global Non-Causal Masking:** Enables attention heads to cross-examine relationships both backward and forward across the entire $3\text{--}5\text{ second}$ timeline simultaneously, isolating the biphase transition from the initial rigid tonic freeze to clonic contractions.
*   **Absolute Positional Embeddings:** Hardcodes chronological time progress into the continuous vector space, tracking explicit postural evolution over the sliding window.
*   **Coordinate Augmentation Layer:** Introduces horizontal spatial mirroring, scale/translation jittering, and random joint tracking dropouts directly inside the training loop to artificially amplify dataset variety.

### Phase 4: Optimization, Training & Java Native Runtime Deployment
The Python script optimizes training infrastructure using Graph Compilation (`torch.compile`) and Automatic Mixed Precision (AMP) with `bfloat16` to maximize local eGPU compute performance. Upon model convergence against validation cross-entropy metrics, the model is exported to an optimized `.onnx` graph. The compiled graph is loaded directly back into the Java runtime environment via the existing ONNX Runtime pipeline. This deep learning classifier evaluates multi-modal vector sequences in real-time alongside low-level FFM pixel-difference fallback frameworks to trigger stable, verified caregiver notifications.
