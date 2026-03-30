# Variational Autoencoder (DAE) – Understanding Latent Space pt. 2

This project is a practical companion to my blog post exploring the intuition behind Deep Learning and the transition from manual feature engineering to automated feature learning. 
The Variational Autoencoder represents the second foundational step on this journey.

Read the [Blog post](https://jani.isohanni.fi/autoencoders-the-foundation-of-generative-ai/) for a detailed explanation of the concepts and the code.

## Concept Overview

An variational Autoencoder (VAE) is a neural network architecture like Autoencoder, but is uses a stochastic step.

In this implementation, we demonstrate how the latent space behaves in this setup and what we need to balance with.

1. Demonstrate the structure of the latent space.
2. Generate few example images.
3. Demonstrate reconstruction capabilities.


## Installation & Usage

Run the code from the project root:
```bash

    python -m deep_learning.vae.vae_mnist
```

