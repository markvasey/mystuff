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
from torch.utils.data import Dataset, DataLoader
from tokenizer import BPETokenizer

# ── MODEL ARCHITECTURE ────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :], persistent=False)

    def forward(self, x):
        T = x.shape[1]
        return self.cos_cached[:, :T, :, :], self.sin_cached[:, :T, :, :]

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, n_kv_head, block_size):
        super().__init__()
        assert d_model % n_head == 0, f"d_model {d_model} must be divisible by n_head {n_head}"
        assert n_head % n_kv_head == 0, f"n_head {n_head} must be divisible by n_kv_head {n_kv_head}"
        
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = d_model // n_head
        self.group_size = n_head // n_kv_head
        self.d_model = d_model
        
        # Projections for GQA (no bias is standard)
        self.q_proj = nn.Linear(d_model, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Rotary positional embeddings
        self.rotary = RotaryEmbedding(self.head_dim, max_position_embeddings=block_size)
        
        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        
        # 1. Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        
        # 2. Apply RoPE
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        # 3. Repeat K and V to match Query heads (Grouped Query Attention)
        if self.group_size > 1:
            k = k.unsqueeze(3).expand(B, T, self.n_kv_head, self.group_size, self.head_dim).reshape(B, T, self.n_head, self.head_dim)
            v = v.unsqueeze(3).expand(B, T, self.n_kv_head, self.group_size, self.head_dim).reshape(B, T, self.n_head, self.head_dim)
            
        # 4. Standard scaled dot-product attention
        q = q.transpose(1, 2)  # [B, n_head, T, head_dim]
        k = k.transpose(1, 2)  # [B, n_head, T, head_dim]
        v = v.transpose(1, 2)  # [B, n_head, T, head_dim]
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v  # [B, n_head, T, head_dim]
        
        # 5. Concatenate and project back
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # SwiGLU uses gate (w1) and value (w2) projections, then projects back (w3).
        # Bias is traditionally False in SwiGLU.
        self.w1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.w2 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.w3 = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x):
        # Swish(x * w1) * (x * w2) where Swish is F.silu in PyTorch
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, d_model, n_head, n_kv_head, block_size):
        super().__init__()
        self.ln_1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, n_kv_head, block_size)
        self.ln_2 = RMSNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_kv_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, n_kv_head, block_size) for _ in range(n_layer)])
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.wte.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is {self.block_size}"
        
        # With RoPE, we don't add absolute positional embeddings
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

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
        chunk = self.token_ids[start_idx:start_idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ── TRAINING & EXPORT RUNNER ──────────────────────────────────────────────────

def clean_gutenberg(text):
    # Match Java's extractBookText logic roughly
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK"
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK"
    ]
    
    start_idx = 0
    for marker in start_markers:
        pos = text.find(marker)
        if pos != -1:
            # Skip until the end of line containing marker
            line_end = text.find("\n", pos)
            if line_end != -1:
                start_idx = line_end + 1
            else:
                start_idx = pos + len(marker)
            break
            
    end_idx = len(text)
    for marker in end_markers:
        pos = text.find(marker)
        if pos != -1:
            end_idx = pos
            break
            
    if start_idx < end_idx:
        return text[start_idx:end_idx].strip()
    return text.strip()

def clean_corpus_text(text):
    # First apply standard Gutenberg header/footer cleaning
    cleaned = clean_gutenberg(text)
    
    # Process line-by-line
    lines = cleaned.split("\n")
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # 1. Filter out page numbers (e.g., "12", "Page 45", "[Page 12]", "PAGE 12")
        if re.match(r"^\[?Page\s+\d+\]?$", stripped, re.IGNORECASE) or re.match(r"^\d+$", stripped):
            continue
            
        # 2. Filter out Roman numeral / chapter/act/scene headings (e.g., "CHAPTER XXIV", "ACT I", "SCENE II")
        if re.match(r"^(chapter|act|scene)\s+[ivxldm\d]+$", stripped, re.IGNORECASE):
            continue
            
        # 3. Filter out transcriber notes or metadata lines
        if "transcriber's note" in stripped.lower() or "produced by" in stripped.lower():
            continue
            
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)

def main():
    parser = argparse.ArgumentParser(description="Train Causal Transformer and Export to ONNX")
    parser.add_argument("--training_dir", type=str, default="Training/TinyStories", help="Corpus files directory")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.bin", help="Path to save/load tokenizer")
    parser.add_argument("--target_vocab_size", type=int, default=4096, help="BPE vocabulary size")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_kv_head", type=int, default=2, help="Number of GQA Key/Value heads")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--block_size", type=int, default=256, help="Sequence block size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--stride", type=int, default=1, help="Stride step size to slice the corpus")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (number of validation-worse epochs before stopping)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--export_onnx", type=str, default="model.onnx", help="Export path for ONNX model")
    args = parser.parse_args()

    # 1. Setup tokenizer
    tokenizer = BPETokenizer()
    if os.path.exists(args.tokenizer_path):
        print(f"Loading BPE Tokenizer from {args.tokenizer_path}...")
        tokenizer.load(args.tokenizer_path)
    else:
        # Train tokenizer
        print(f"Scanning training corpus directory: {args.training_dir}")
        txt_files = [os.path.join(args.training_dir, f) for f in os.listdir(args.training_dir) if f.endswith(".txt")]
        if not txt_files:
            print(f"Error: No text files found in {args.training_dir}")
            sys.exit(1)
        
        corpus_parts = []
        for path in txt_files:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                corpus_parts.append(clean_corpus_text(content))
        full_corpus = "\n\n".join(corpus_parts)
        
        tokenizer.train(full_corpus, args.target_vocab_size)
        tokenizer.save(args.tokenizer_path)
        print(f"BPE Tokenizer trained and saved to {args.tokenizer_path}")

    # 2. Tokenize corpus for training (or load from cache if available)
    tokens_cache = "tokens.bin"
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
                corpus_parts.append(clean_corpus_text(content))
        full_corpus = "\n\n".join(corpus_parts)
        
        token_ids = tokenizer.encode(full_corpus)
        # Save to cache
        np.array(token_ids, dtype=np.int32).tofile(tokens_cache)
        print(f"Saved tokenized corpus to cache ({tokens_cache})")
        
    print(f"Total tokens in corpus: {len(token_ids)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # 3. Create datasets & dataloaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Enable TF32 Tensor Cores for any remaining FP32 matmuls on Ampere/Ada Lovelace
    if device.type == "cuda":
        torch.set_float32_matmul_precision('high')
        
    dataset = CorpusDataset(token_ids, args.block_size, stride=args.stride)
    # 90/10 train/validation split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    use_cuda = (device.type == "cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=use_cuda,
        num_workers=2 if use_cuda else 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=use_cuda,
        num_workers=2 if use_cuda else 0
    )
    
    # 4. Initialize model
    model = CausalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_layer=args.n_layer,
        block_size=args.block_size
    ).to(device)

    # Count parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")

    # Compile the model for PyTorch 2.x optimized JIT execution (Kernel Fusion + Triton)
    if device.type == "cuda":
        print("Compiling model graph using torch.compile(mode='reduce-overhead')...")
        compiled_model = torch.compile(model, mode="reduce-overhead")
    else:
        compiled_model = model

    def get_lr(step, max_steps, warmup_steps, lr_max, lr_min):
        # 1. Linear warmup
        if step < warmup_steps:
            return lr_max * step / max(1, warmup_steps)
        # 2. Cosine decay
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min + cos_decay * (lr_max - lr_min)

    # 5. Optimizer & loss
    # Filter parameters for weight decay exclusion
    decay = set()
    no_decay = set()
    
    # Iterate over unique parameters in named_parameters() to avoid duplicates from tied weights (e.g. lm_head.weight and wte.weight)
    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if pn.endswith('bias'):
            # all biases are excluded from weight decay
            no_decay.add(pn)
        elif pn.endswith('weight'):
            # Find the submodule that owns this parameter to check its type
            if '.' in pn:
                submodule_path, _ = pn.rsplit('.', 1)
                m = model.get_submodule(submodule_path)
            else:
                m = model
            
            if isinstance(m, nn.Linear):
                # weights of Linear modules are decayed
                decay.add(pn)
            elif isinstance(m, (nn.LayerNorm, RMSNorm, nn.Embedding)):
                # weights of Norm/Embeddings are excluded from decay
                no_decay.add(pn)

    # Validate that we got all parameters
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not categorized!"

    # Create optimizer groups
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    # Enable fused AdamW on CUDA to run parameter updates inside a single optimized kernel
    use_fused = (device.type == "cuda")
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, fused=use_fused)
    
    checkpoint_path = "checkpoint.pt"
    start_epoch = 1
    best_val_loss = float("inf")
    worse_epochs = 0
    
    if os.path.exists(checkpoint_path):
        print(f"Loading PyTorch checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        worse_epochs = checkpoint.get("worse_epochs", 0)
        print(f"Resuming training from Epoch {start_epoch} (Best Val Loss: {best_val_loss:.4f})")

    # Warmup and Cosine Decay parameters
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.05 * total_steps)  # 5% warmup
    lr_min = args.lr * 0.1  # decay down to 10%

    device_type = "cuda" if "cuda" in str(device) else "cpu"
    enable_amp = (device.type == "cuda")

    # 6. Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0
        start_time = time.time()
        for batch_idx, (x, y) in enumerate(train_loader):
            # Calculate learning rate for this global step and update optimizer
            global_step = (epoch - 1) * len(train_loader) + batch_idx
            lr = get_lr(global_step, total_steps, warmup_steps, args.lr, lr_min)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=enable_amp):
                logits = compiled_model(x)
                # Flatten logits and targets
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                elapsed = time.time() - start_time
                seq_per_sec = (batch_idx * args.batch_size) / elapsed if elapsed > 0 else 0.0
                tok_per_sec = seq_per_sec * args.block_size
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Speed: {seq_per_sec:.1f} seq/s ({tok_per_sec:.0f} tok/s)")

        avg_train_loss = total_loss / len(train_loader)
        
        # Validation pass
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=enable_amp):
                    logits = compiled_model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"--- Epoch {epoch} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {time.time() - start_time:.1f}s ---")
        
        # Generate and print a text sample to visualize training progress
        sample_text = generate_sample(model, tokenizer, "The ", num_tokens=50, device=device)
        print(f"Sample (Epoch {epoch}): [{sample_text}]")
        model.train() # restore train mode

        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            worse_epochs = 0
            print("✓ New best validation loss. Saving checkpoint...")
            # Save PyTorch checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "worse_epochs": worse_epochs
            }, checkpoint_path)
        else:
            worse_epochs += 1
            print(f"⚠ Validation loss did not improve ({worse_epochs}/{args.patience} patience). Best: {best_val_loss:.4f}")
            if worse_epochs >= args.patience:
                print(f"Early stopping triggered at Epoch {epoch}. Training stopped.")
                break

    print("Training finished!")
    
    # Export the final best model to ONNX at the very end of training
    if os.path.exists(checkpoint_path):
        print("Loading best checkpoint for final ONNX export...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    print(f"Exporting final optimized model to {args.export_onnx}...")
    export_onnx(model, args.export_onnx, args.block_size, device)

def export_onnx(model, onnx_path, block_size, device):
    model.eval()
    dummy_input = torch.zeros((1, block_size), dtype=torch.long, device=device)
    
    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        }
    )
    print(f"Model successfully exported to {onnx_path}")

def generate_sample(model, tokenizer, prompt, num_tokens=50, temperature=0.7, top_k=5, device="cpu"):
    model.eval()
    token_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(num_tokens):
            cond_input = input_tensor[:, -model.block_size:]
            logits = model(cond_input)
            last_logits = logits[0, -1, :] / max(temperature, 1e-6)
            
            # Apply top-k filtering
            v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
            last_logits[last_logits < v[-1]] = -float('inf')
            
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_tensor = torch.cat((input_tensor, next_token.unsqueeze(0)), dim=1)
            
    generated_ids = input_tensor[0].tolist()
    return tokenizer.decode(generated_ids)

if __name__ == "__main__":
    main()
