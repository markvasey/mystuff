import os
import sys
import time
import math
import struct
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizer import BPETokenizer

# ── MODEL ARCHITECTURE ────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, block_size):
        super().__init__()
        assert d_model % n_head == 0, f"d_model {d_model} must be divisible by n_head {n_head}"
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.n_head = n_head
        self.d_model = d_model
        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_head, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights (optional, standard in transformer models)
        self.wte.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is {self.block_size}"
        
        # Position index
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        x = self.wte(idx) + self.wpe(pos)
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

def main():
    parser = argparse.ArgumentParser(description="Train Causal Transformer and Export to ONNX")
    parser.add_argument("--training_dir", type=str, default="Training/TinyStories", help="Corpus files directory")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.bin", help="Path to save/load tokenizer")
    parser.add_argument("--target_vocab_size", type=int, default=4096, help="BPE vocabulary size")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--block_size", type=int, default=256, help="Sequence block size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--stride", type=int, default=1, help="Stride step size to slice the corpus")
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
                corpus_parts.append(clean_gutenberg(content))
        full_corpus = "\n\n".join(corpus_parts)
        
        tokenizer.train(full_corpus, args.target_vocab_size)
        tokenizer.save(args.tokenizer_path)
        print(f"BPE Tokenizer trained and saved to {args.tokenizer_path}")

    # 2. Tokenize corpus for training
    print("Tokenizing corpus for training...")
    txt_files = [os.path.join(args.training_dir, f) for f in os.listdir(args.training_dir) if f.endswith(".txt")]
    corpus_parts = []
    for path in txt_files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            corpus_parts.append(clean_gutenberg(content))
    full_corpus = "\n\n".join(corpus_parts)
    
    token_ids = tokenizer.encode(full_corpus)
    print(f"Total tokens in corpus: {len(token_ids)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # 3. Create datasets & dataloaders
    dataset = CorpusDataset(token_ids, args.block_size, stride=args.stride)
    # 90/10 train/validation split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    # 4. Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model = CausalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block_size
    ).to(device)

    # Count parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")

    # 5. Optimizer & loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # 6. Training loop
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        start_time = time.time()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            # Flatten logits and targets
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Speed: {batch_idx * args.batch_size / elapsed:.1f} tok/s")

        avg_train_loss = total_loss / len(train_loader)
        
        # Validation pass
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"--- Epoch {epoch} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {time.time() - start_time:.1f}s ---")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("✓ New best validation loss. Exporting checkpoints to ONNX...")
            # Export to ONNX
            export_onnx(model, args.export_onnx, args.block_size, device)

    print("Training finished!")

def export_onnx(model, onnx_path, block_size, device):
    model.eval()
    dummy_input = torch.zeros((1, block_size), dtype=torch.long, device=device)
    
    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        }
    )
    print(f"Model successfully exported to {onnx_path}")

if __name__ == "__main__":
    main()
