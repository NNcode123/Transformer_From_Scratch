# Transformer_From_Scratch

A native PyTorch implementation of a transformer-style architecture built from scratch for learning and experimentation.

## Project Intent

This repository is organized as a small research-friendly codebase for understanding the major pieces of modern sequence modeling:

- Data preprocessing and tokenization live under the data folder.
- The Model folder contains core building blocks such as embeddings, attention, encoder/decoder layers, a full transformer, and a sparse autoencoder module.
- The train folder provides simple entry points for training, testing, and evaluation.
- The utils folder stores configuration values such as hyperparameters in YAML.

## Repository Structure

- data/
  - dataloader.py: basic dataset and batch-loading utilities
  - tokenizer.py: a simple tokenizer for turning text into integer IDs
- Model/
  - transformer.py: full encoder-decoder transformer
  - encoder_block.py: encoder block with self-attention and feed-forward layers
  - decoder_block.py: decoder block with self-attention, cross-attention, and feed-forward layers
  - embedding.py: token embedding layer
  - Attention.py: multi-head attention implementation
  - sparse_auto_encoder.py: lightweight sparse autoencoder module for representation learning
- train/
  - train.py: training entry point
  - test.py: testing entry point
  - evaluate.py: evaluation entry point
- utils/
  - hyperparameters.yaml: default model and training settings

## Intended Usage

This project is intended as a learning scaffold for:

1. Building transformer components by hand.
2. Experimenting with tokenization and data loading.
3. Extending the architecture with new modules such as sparse autoencoders or additional training logic.

You can start by inspecting the modules in the Model folder and then wiring them together in the training pipeline.
