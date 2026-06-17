import os
import sys
import time
import math
import struct
import argparse
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tokenizer import BPETokenizer
import transformers
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)

# ── DATA LOADING ──────────────────────────────────────────────────────────────

class CorpusDataset(Dataset):
    def __init__(self, token_ids, block_size, stride=1):
        self.token_ids = token_ids
        self.block_size = block_size
        self.stride = stride
        self.start_indices = list(range(0, len(token_ids) - block_size, stride))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_idx = self.start_indices[idx]
        chunk = self.token_ids[start_idx:start_idx + self.block_size]
        x = torch.tensor(chunk, dtype=torch.long)
        # Hugging Face causal LM expects labels to be equal to input_ids;
        # it handles shifting internally in LlamaForCausalLM
        return {"input_ids": x, "labels": x}

# ── DATA CLEANING ─────────────────────────────────────────────────────────────
# Gutenberg cleaning functions removed since TinyStories is clean.

# ── SAMPLING & LOGGING ────────────────────────────────────────────────────────

def generate_sample(model, tokenizer, prompt="The ", max_len=60, temp=0.8, top_k=50):
    model.eval()
    device = next(model.parameters()).device
    
    # Extract the base model if wrapped by torch.compile (which wraps it in _CompiledModule)
    raw_model = model
    if hasattr(model, "_orig_mod"):
        raw_model = model._orig_mod
        
    x = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_len):
        x_cond = x[:, -1024:]
        with torch.no_grad():
            outputs = raw_model(x_cond)
            logits = outputs.logits
        logits = logits[:, -1, :] / temp
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)
        if next_token.item() == 2:  # EOS token
            break
    model.train()
    return tokenizer.decode(x[0].tolist())

class CustomLoggingCallback(TrainerCallback):
    def __init__(self, tokenizer, num_batches_per_epoch):
        self.tokenizer = tokenizer
        self.num_batches_per_epoch = num_batches_per_epoch

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            loss = logs["loss"]
            lr = logs.get("learning_rate", 0.0)
            epoch = math.floor(state.epoch) + 1
            global_step = state.global_step
            batch_idx = (global_step - 1) % self.num_batches_per_epoch + 1
            print(f"Epoch {epoch} | Batch {batch_idx}/{self.num_batches_per_epoch} | Loss: {loss:.4f} | LR: {lr:.2e}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        eval_loss = metrics.get("eval_loss", None)
        if eval_loss is not None:
            epoch = math.floor(state.epoch)
            print(f"--- Epoch {epoch} Summary | Val Loss: {eval_loss:.4f} ---")

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        sample_text = generate_sample(model, self.tokenizer)
        epoch = math.floor(state.epoch)
        print(f"Sample (Epoch {epoch}): [{sample_text}]")

# ── ONNX EXPORT WRAPPER ───────────────────────────────────────────────────────

class ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        # Passes input_ids directly. LlamaForCausalLM automatically
        # generates attention mask and position IDs internally.
        outputs = self.model(input_ids=input_ids)
        return outputs.logits

# ── MAIN RUNNER ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Hugging Face LLaMA and Export to ONNX")
    parser.add_argument("--training_dir", type=str, default="Training/TinyStories_200k", help="Corpus files directory")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.bin", help="Path to save/load tokenizer")
    parser.add_argument("--target_vocab_size", type=int, default=4096, help="BPE vocabulary size")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension (hidden_size)")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_kv_head", type=int, default=2, help="Number of GQA Key/Value heads")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--block_size", type=int, default=256, help="Sequence block size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--stride", type=int, default=1, help="Stride step size to slice the corpus")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--export_onnx", type=str, default="model.onnx", help="Export path for ONNX model")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile JIT compilation")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps to limit training")
    args = parser.parse_args()

    # 1. Setup tokenizer
    tokenizer = BPETokenizer()
    if os.path.exists(args.tokenizer_path):
        print(f"Loading BPE Tokenizer from {args.tokenizer_path}...")
        tokenizer.load(args.tokenizer_path)
    else:
        print(f"Scanning training corpus directory: {args.training_dir}")
        txt_files = [os.path.join(args.training_dir, f) for f in os.listdir(args.training_dir) if f.endswith(".txt")]
        if not txt_files:
            print(f"Error: No text files found in {args.training_dir}")
            sys.exit(1)
        
        corpus_parts = []
        for path in txt_files:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                corpus_parts.append(content.strip())
        full_corpus = "\n\n".join(corpus_parts)
        
        tokenizer.train(full_corpus, args.target_vocab_size)
        tokenizer.save(args.tokenizer_path)
        print(f"BPE Tokenizer trained and saved to {args.tokenizer_path}")

    # 2. Tokenize corpus for training (or load from cache if available)
    # Use cache name linked to tokenizer path to avoid vocab mismatch crashes
    tokens_cache = args.tokenizer_path.replace(".bin", "_tokens.bin")
    if tokens_cache == "tokenizer_tokens.bin":
        tokens_cache = "tokens.bin"  # Keep backward compatibility with standard name
    if os.path.exists(tokens_cache):
        print(f"Loading tokenized corpus from cache ({tokens_cache})...")
        token_ids = np.fromfile(tokens_cache, dtype=np.int32).tolist()
    else:
        print("Tokenizing corpus for training...")
        txt_files = [os.path.join(args.training_dir, f) for f in os.listdir(args.training_dir) if f.endswith(".txt")]
        corpus_parts = []
        for path in txt_files:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                corpus_parts.append(content.strip())
        full_corpus = "\n\n".join(corpus_parts)
        
        token_ids = tokenizer.encode(full_corpus)
        np.array(token_ids, dtype=np.int32).tofile(tokens_cache)
        print(f"Saved tokenized corpus to cache ({tokens_cache})")
        
    print(f"Total tokens in corpus: {len(token_ids)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # 3. Create datasets & dataloaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    if device.type == "cuda":
        torch.set_float32_matmul_precision('high')
        
    dataset = CorpusDataset(token_ids, args.block_size, stride=args.stride)
    
    # 90/10 train/validation split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Calculate batches per epoch
    num_batches_per_epoch = len(train_dataset) // args.batch_size
    print(f"Train Dataset Size: {len(train_dataset)} sequences ({num_batches_per_epoch} batches/epoch)")
    print(f"Val Dataset Size: {len(val_dataset)} sequences")

    # 4. Initialize LLaMA Model Configuration
    # Using LlamaConfig ensures GQA, RoPE, RMSNorm, and SwiGLU are configured natively
    llama_config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.d_model,
        intermediate_size=4 * args.d_model,  # intermediate dim matches 4 * hidden_size SwiGLU projection
        num_attention_heads=args.n_head,
        num_key_value_heads=args.n_kv_head,
        num_hidden_layers=args.n_layer,
        max_position_embeddings=args.block_size,
        rms_norm_eps=1e-6,
        attention_bias=False,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    
    model = LlamaForCausalLM(llama_config).to(device)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")

    # Compile the model if requested
    if device.type == "cuda" and not args.no_compile:
        print("Compiling model graph using torch.compile(mode='reduce-overhead')...")
        model = torch.compile(model, mode="reduce-overhead")

    # 5. Set up Training Arguments
    use_bf16 = (device.type == "cuda")
    
    # Auto-resume from latest checkpoint if checkpoints exist
    resume_path = None
    output_dir = "./checkpoints"
    if os.path.exists(output_dir):
        dirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if dirs:
            dirs.sort(key=lambda x: int(x.split("-")[-1]))
            resume_path = dirs[-1]
            print(f"Found existing checkpoint. Will resume from: {resume_path}")

    is_test_run = (args.max_steps > 0)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=1 if is_test_run else 100,
        eval_strategy="no" if is_test_run else "epoch",
        save_strategy="no" if is_test_run else "epoch",
        save_total_limit=2,
        bf16=use_bf16,
        fp16=False,
        dataloader_pin_memory=use_bf16,
        dataloader_num_workers=0,  # 0 prevents multiprocessing worker hangs on Python 3.14
        report_to="none",
        load_best_model_at_end=not is_test_run,
        metric_for_best_model="eval_loss" if not is_test_run else None,
        greater_is_better=False,
        remove_unused_columns=False,
    )

    # 6. Initialize Trainer
    callbacks = [CustomLoggingCallback(tokenizer, num_batches_per_epoch)]
    if not is_test_run:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks,
    )

    # Disable default console logging callbacks to prevent clashing print statements
    try:
        from transformers.trainer_callback import PrinterCallback, ProgressCallback
        trainer.remove_callback(PrinterCallback)
        trainer.remove_callback(ProgressCallback)
    except Exception:
        pass

    # 7. Start Training
    print("Starting Hugging Face LLaMA model training...")
    trainer.train(resume_from_checkpoint=resume_path)
    print("Training complete!")

    # 8. Load best checkpoint and export to ONNX
    if trainer.state.best_model_checkpoint is not None:
        print(f"Loading best model checkpoint from {trainer.state.best_model_checkpoint}...")
        best_model = LlamaForCausalLM.from_pretrained(trainer.state.best_model_checkpoint)
    else:
        best_model = model
        if hasattr(best_model, "_orig_mod"):
            best_model = best_model._orig_mod

    # Export to ONNX on CPU
    best_model = best_model.cpu()
    onnx_model = ONNXWrapper(best_model)
    onnx_model.eval()

    dummy_input = torch.ones((1, 8), dtype=torch.long)
    print(f"Exporting final optimized model to {args.export_onnx}...")
    
    torch.onnx.export(
        onnx_model,
        dummy_input,
        args.export_onnx,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=18
    )
    print("ONNX model successfully exported!")

if __name__ == "__main__":
    main()
