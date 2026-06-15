#!/bin/bash
venv/bin/python train.py \
  --training_dir Training/TinyStories \
  --tokenizer_path tokenizer.bin \
  --target_vocab_size 8192 \
  --d_model 512 \
  --n_head 8 \
  --n_kv_head 2 \
  --n_layer 12 \
  --block_size 1024 \
  --stride 512 \
  --epochs 80 \
  --batch_size 8 \
  --lr 0.0003 \
  --export_onnx model.onnx

