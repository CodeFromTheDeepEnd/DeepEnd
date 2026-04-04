# Building Controllable Latent Representations with Typed Autoencoders

This project is a practical companion to my blog post exploring the intuition behind Deep Learning 
and the transition from manual feature engineering to automated feature learning. 
The "Typed Autoencoder" represents how to control the latent space variables, and how to
place clear semantics in the dimension.

Read the [Blog post](https://jani.isohanni.fi/autoencoders-the-foundation-of-generative-ai/) for a detailed explanation of the concepts and the code.

## Concept Overview

A Typed Autoencoder extends the standard autoencoder by assigning semantic meaning to specific latent dimensions.
Instead of letting the network freely organize the latent space, we explicitly reserve coordinates for controllable
attributes (like rotation) while allowing others to capture remaining variance.

This implementation demonstrates:

* 3-Phase Training Strategy: Freeze encoder → train decoder → align both for stable semantic learning
* Explicit Semantic Control: Reserve one latent dimension for rotation angle (0-360°)
* Controllable Generation: Generate digits at any specified rotation angle
* Latent Space Visualization: Explore how semantic and free dimensions organize information

The approach works well on MNIST (digits) but shows limitations on Fashion-MNIST, illustrating when explicit
semantic control is feasible versus when disentanglement techniques like β-VAE might be needed.

## Installation & Usage

Run the code from the project root:
```bash
    # Train the typed autoencoder on MNIST
    python -m deep_learning.typed_ae.train_rot_mnist

    # Test and visualize the results
    python -m deep_learning.typed_ae.test_rot_mnist
```

## Results

The trained model can:

* Generate MNIST digits at any specified rotation angle
* Reconstruct rotated digits while preserving rotation information
* Demonstrate how explicit semantic coordinates affect latent space organization

See the blog post for detailed results, visualizations, and discussion of when this approach works versus its limitations.