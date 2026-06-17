import os
import json
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)

# ── DATA LOADING & AUGMENTATION ───────────────────────────────────────────────

class SkeletalDataset(Dataset):
    def __init__(self, data_dir, seq_len=32, stride=4, augment=False):
        self.seq_len = seq_len
        self.stride = stride
        self.augment = augment
        self.samples = []

        print(f"Scanning for JSON coordinate files in: {data_dir}...")
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        label = data['label']
                        features = [f['features'] for f in data['sequence']]
                        
                        if len(features) < seq_len:
                            continue
                        
                        features_arr = np.array(features, dtype=np.float32)
                        
                        # Use a short step (stride) for training, and full non-overlapping window sizes for validation
                        step = stride if augment else seq_len
                        for start_idx in range(0, len(features_arr) - seq_len + 1, step):
                            window = features_arr[start_idx:start_idx + seq_len]
                            self.samples.append((window, label))
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

        print(f"Loaded {len(self.samples)} sample windows of length {seq_len} from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window, label = self.samples[idx]
        window = window.copy()

        if self.augment:
            window = self.apply_augmentations(window)

        return {
            "input_features": torch.tensor(window, dtype=torch.float32),
            "labels": torch.tensor(label, dtype=torch.long)
        }

    def apply_augmentations(self, window):
        # 1. Horizontal Mirroring (50% probability)
        if np.random.rand() > 0.5:
            # Invert X coordinate values (feature index 3*i is x, 3*i+1 is y, 3*i+2 is confidence)
            for i in range(17):
                window[:, 3 * i] = 1.0 - window[:, 3 * i]

            # Swap left-right joint indices to maintain anatomical correctness
            left_right_pairs = [
                (1, 2),   # Eyes
                (3, 4),   # Ears
                (5, 6),   # Shoulders
                (7, 8),   # Elbows
                (9, 10),  # Wrists
                (11, 12), # Hips
                (13, 14), # Knees
                (15, 16)  # Ankles
            ]
            for left, right in left_right_pairs:
                temp = window[:, 3 * left : 3 * left + 3].copy()
                window[:, 3 * left : 3 * left + 3] = window[:, 3 * right : 3 * right + 3]
                window[:, 3 * right : 3 * right + 3] = temp

        # 2. Scale Jittering (random scale factor in range [0.95, 1.05])
        scale = np.random.uniform(0.95, 1.05)
        for i in range(17):
            window[:, 3 * i] *= scale
            window[:, 3 * i + 1] *= scale

        # 3. Translation Jittering (random coordinates offset in range [-0.05, 0.05])
        shift_x = np.random.uniform(-0.05, 0.05)
        shift_y = np.random.uniform(-0.05, 0.05)
        for i in range(17):
            window[:, 3 * i] += shift_x
            window[:, 3 * i + 1] += shift_y

        # 4. Joint Dropout (5% probability of losing individual joint tracking)
        for i in range(17):
            if np.random.rand() < 0.05:
                window[:, 3 * i : 3 * i + 3] = 0.0

        # Clip values to ensure coordinates remain strictly inside bounding dimensions
        for i in range(17):
            window[:, 3 * i] = np.clip(window[:, 3 * i], 0.0, 1.0)
            window[:, 3 * i + 1] = np.clip(window[:, 3 * i + 1], 0.0, 1.0)

        return window

# ── MODEL ARCHITECTURE ────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SeizureTransformer(nn.Module):
    def __init__(self, feature_dim=51, d_model=64, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.feature_projection = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, input_features, labels=None, **kwargs):
        # input_features shape: [batch_size, seq_len, feature_dim]
        x = self.feature_projection(input_features) # [batch_size, seq_len, d_model]
        x = self.pos_encoder(x)
        x = self.transformer(x)                     # [batch_size, seq_len, d_model]
        
        # Bidirectional global temporal pooling
        x = x.mean(dim=1)                           # [batch_size, d_model]
        logits = self.classifier(x)                 # [batch_size, num_classes]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return {"loss": loss, "logits": logits}

# ── ONNX EXPORT WRAPPER ───────────────────────────────────────────────────────

class ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features):
        outputs = self.model(input_features=input_features)
        # Apply Softmax to get probabilities for Java runtime
        return nn.functional.softmax(outputs["logits"], dim=-1)

# ── LOGGING CALLBACK ──────────────────────────────────────────────────────────

class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            loss = logs["loss"]
            lr = logs.get("learning_rate", 0.0)
            epoch = state.epoch
            print(f"Epoch {epoch:.2f} | Step {state.global_step} | Loss: {loss:.4f} | LR: {lr:.2e}")

# ── MAIN RUNNER ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Spatio-Temporal Seizure Detection Transformer")
    parser.add_argument("--train_dir", type=str, default="TestVideos", help="Path to test videos parent directory containing JSON data")
    parser.add_argument("--d_model", type=int, default=64, help="Model embedding dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of Transformer layers")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence context window length")
    parser.add_argument("--stride", type=int, default=4, help="Slicing stride for training window overlaps")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--export_onnx", type=str, default="seizure_transformer.onnx", help="Export path for ONNX model")
    args = parser.parse_args()

    # 1. Load splits
    train_dataset = SkeletalDataset(args.train_dir, seq_len=args.seq_len, stride=args.stride, augment=True)
    val_dataset = SkeletalDataset(args.train_dir, seq_len=args.seq_len, stride=args.seq_len, augment=False)

    if len(train_dataset) == 0:
        print("Error: No training samples loaded. Ensure DatasetExtractor has run and generated JSON files.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Initialize Model
    model = SeizureTransformer(
        feature_dim=51, # 17 joints * 3
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        num_classes=2
    ).to(device)

    # Compile the model for JIT graph optimization
    if device.type == "cuda":
        print("Compiling model graph using torch.compile(mode='reduce-overhead')...")
        model = torch.compile(model, mode="reduce-overhead")

    # 3. Setup Training Arguments
    use_bf16 = (device.type == "cuda")
    training_args = TrainingArguments(
        output_dir="./transformer_checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=use_bf16,
        fp16=False,
        dataloader_pin_memory=use_bf16,
        dataloader_num_workers=0,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"]
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[CustomLoggingCallback(), EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Disable default console logging callbacks to prevent duplicate text
    try:
        from transformers.trainer_callback import PrinterCallback, ProgressCallback
        trainer.remove_callback(PrinterCallback)
        trainer.remove_callback(ProgressCallback)
    except Exception:
        pass

    # 5. Train
    print("Starting Spatio-Temporal Transformer training...")
    trainer.train()
    print("Training complete!")

    # 6. Load best model checkpoint and export to ONNX
    if trainer.state.best_model_checkpoint is not None:
        print(f"Loading best model checkpoint from {trainer.state.best_model_checkpoint}...")
        best_model = SeizureTransformer(
            feature_dim=51,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            num_classes=2
        )
        # Load state dict (safetensors or pytorch bin)
        safetensors_path = os.path.join(trainer.state.best_model_checkpoint, "model.safetensors")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
        else:
            state_dict = torch.load(os.path.join(trainer.state.best_model_checkpoint, "pytorch_model.bin"))
            
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        best_model.load_state_dict(clean_state_dict)
    else:
        best_model = model
        if hasattr(best_model, "_orig_mod"):
            best_model = best_model._orig_mod

    # Export wrapper to CPU
    best_model = best_model.cpu()
    onnx_model = ONNXWrapper(best_model)
    onnx_model.eval()

    dummy_input = torch.ones((1, args.seq_len, 51), dtype=torch.float32)
    print(f"Exporting final optimized model to {args.export_onnx}...")
    
    torch.onnx.export(
        onnx_model,
        dummy_input,
        args.export_onnx,
        input_names=["input_features"],
        output_names=["probabilities"],
        dynamic_axes={
            "input_features": {0: "batch_size", 1: "sequence_length"},
            "probabilities": {0: "batch_size"}
        },
        opset_version=17
    )
    print("ONNX model successfully exported!")

if __name__ == "__main__":
    main()
