#!/bin/bash
# Remove existing checkpoints to force training from scratch
rm -f model.bin model_old.bin model_double.bin

# Run the training script
./train_model.sh
