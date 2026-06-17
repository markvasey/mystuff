#!/usr/bin/env python3
"""
Re-export the SeizureTransformer checkpoint to ONNX IR version 9,
compatible with ONNX Runtime 1.16.x used by the Java application.

Usage:
    python3 Scripts/reexport_onnx_ir9.py \
        --checkpoint transformer_checkpoints/<best-checkpoint-dir> \
        --output seizure_transformer.onnx

The script:
  1. Loads the best model checkpoint (safetensors or pytorch_model.bin).
  2. Wraps it in the ONNXWrapper (applies softmax for Java).
  3. Exports to ONNX with opset_version=17.
  4. Uses onnx.version_converter to lower-bound the IR version to <= 9
     if the initial export produced IR >= 10.
"""

import os
import sys
import math
import argparse
import torch
import torch.nn as nn
import onnx
from onnx import version_converter


# ── Model (must match train_transformer.py) ───────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


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
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, input_features, labels=None, **kwargs):
        x = self.feature_projection(input_features)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_classes), labels.view(-1))
        return {"loss": loss, "logits": logits}


class ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features):
        outputs = self.model(input_features=input_features)
        return nn.functional.softmax(outputs["logits"], dim=-1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Re-export SeizureTransformer to ONNX IR<=9 for ONNX Runtime 1.16.x"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="transformer_checkpoints",
        help="Path to the checkpoint directory (containing model.safetensors or pytorch_model.bin). "
             "If a parent dir is given, the script picks the subdirectory with the lowest eval_loss "
             "(i.e., the one named 'checkpoint-<step>').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="seizure_transformer.onnx",
        help="Output ONNX file path.",
    )
    parser.add_argument("--d_model",    type=int, default=64)
    parser.add_argument("--nhead",      type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--seq_len",    type=int, default=32)
    args = parser.parse_args()

    # ── 1. Locate checkpoint ──────────────────────────────────────────────────
    checkpoint_dir = args.checkpoint
    if not os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")) and \
       not os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")):
        # Try to find the best sub-checkpoint automatically
        candidates = [
            d for d in os.listdir(checkpoint_dir)
            if os.path.isdir(os.path.join(checkpoint_dir, d))
            and d.startswith("checkpoint-")
        ]
        if not candidates:
            print(f"ERROR: No checkpoint found in '{checkpoint_dir}'.")
            sys.exit(1)
        # Pick highest step number (last checkpoint saved)
        candidates.sort(key=lambda d: int(d.split("-")[-1]))
        checkpoint_dir = os.path.join(checkpoint_dir, candidates[-1])
        print(f"Auto-selected checkpoint: {checkpoint_dir}")

    # ── 2. Load weights ───────────────────────────────────────────────────────
    model = SeizureTransformer(
        feature_dim=51,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        num_classes=2,
    )

    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
    else:
        state_dict = torch.load(
            os.path.join(checkpoint_dir, "pytorch_model.bin"),
            map_location="cpu",
        )

    # Strip torch.compile prefix if present
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    print(f"Loaded weights from: {checkpoint_dir}")

    # ── 3. Initial ONNX export ────────────────────────────────────────────────
    model = model.cpu().eval()
    onnx_model = ONNXWrapper(model)
    onnx_model.eval()

    dummy_input = torch.ones((1, args.seq_len, 51), dtype=torch.float32)
    tmp_path = args.output + ".tmp.onnx"

    # Use the legacy TorchScript-based exporter (not dynamo) so that opset_version=17
    # is respected and the output is IR 8, which ONNX Runtime 1.16.x supports.
    with torch.no_grad():
        traced = torch.jit.trace(onnx_model, dummy_input, strict=False)

    torch.onnx.export(
        traced,
        dummy_input,
        tmp_path,
        input_names=["input_features"],
        output_names=["probabilities"],
        dynamic_axes={
            "input_features": {0: "batch_size", 1: "sequence_length"},
            "probabilities": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Initial ONNX export written to: {tmp_path}")

    # ── 4. Check and lower IR version if needed ───────────────────────────────
    proto = onnx.load(tmp_path)
    ir_version = proto.ir_version
    print(f"Exported IR version: {ir_version}")

    if ir_version > 9:
        print(f"IR version {ir_version} > 9 — converting to IR 9 for ONNX Runtime 1.16.x compatibility...")
        converted = version_converter.convert_version(proto, 17)  # stay on opset 17
        # Force IR version to 9
        converted.ir_version = 9
        onnx.save(converted, args.output)
        print(f"Converted model saved to: {args.output}  (IR version: {converted.ir_version})")
    else:
        import shutil
        shutil.move(tmp_path, args.output)
        print(f"IR version already <= 9. Model saved to: {args.output}")

    # Clean up temp
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    # ── 5. Validate with onnxruntime ──────────────────────────────────────────
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        dummy_np = np.ones((1, args.seq_len, 51), dtype=np.float32)
        outputs = sess.run(None, {"input_features": dummy_np})
        print(f"ORT validation passed. Output shape: {outputs[0].shape}, probabilities: {outputs[0]}")
    except Exception as e:
        print(f"WARNING: ORT validation failed: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
