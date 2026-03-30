# Denoising Autoencoder (DAE) – Understanding Latent Space

This project is a practical companion to my blog post exploring the intuition behind Deep Learning and the transition from manual feature engineering to automated feature learning. 
The Autoencoder represents the first foundational step on this journey.

Read the [Blog post](https://jani.isohanni.fi/autoencoders-the-foundation-of-generative-ai/) for a detailed explanation of the concepts and the code.

## Concept Overview

An Autoencoder (AE) is a neural network architecture designed to compress input data (like an image) into a lower-dimensional latent space (the bottleneck) and then reconstruct the original input from this representation.

In this implementation, we focus on a Denoising Autoencoder and demonstrate the idea behind latent space:

1. We feed the network "noisy" or corrupted versions of MNIST digits.
2. The network is forced to squeeze the essential information through a narrow bottleneck.
3. The decoder reconstructs a clean version of the original image based solely on this "core essence."

This process proves that the network isn't just copying pixels—it's learning to understand the shapes and structures of the digits (self-supervised learning).

## Features & Visualizations

The included script (ae_denoise.py) generates four distinct visualizations to help peek into the model's "mind":

1. Denoising Performance: Showcases how the model removes artificial noise from various digits.
2. Reconstruction: Compares clean original images with the model's regenerated versions.
3. Latent Space Clustering (UMAP): Visualizes the 32-dimensional latent space projected onto a 2D plane. You can see how the network naturally groups similar digits together without ever being told their labels.
4. Noise Stress Test: Tests at what point the noise level becomes too high for the model's "intuition" to reconstruct the digit correctly.

## Installation & Usage

Run the code from the project root:
```bash

    python -m deep_learning.autoencoder.ae_denoise
```

