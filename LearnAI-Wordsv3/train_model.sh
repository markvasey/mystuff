#!/bin/bash
venv/bin/python train.py \
  --training_dir Training/FictionalLiterature \
  --tokenizer_path tokenizer.bin \
  --target_vocab_size 4096 \
  --d_model 256 \
  --n_head 4 \
  --n_layer 4 \
  --block_size 256 \
  --stride 10 \
  --epochs 80 \
  --batch_size 64 \
  --lr 0.0003 \
  --export_onnx model.onnx

