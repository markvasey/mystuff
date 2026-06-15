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
  --epochs 10 \
  --batch_size 32 \
  --lr 0.0005 \
  --export_onnx model.onnx

