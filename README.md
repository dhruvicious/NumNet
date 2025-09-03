# numNet 🎲

A small deep learning framework built from scratch with NumPy, made just to see how things work under the hood.

## Overview

numNet is a tiny deep learning framework with a PyTorch-like API, built only with NumPy. It’s a stripped-down implementation of core pieces of deep learning, written just to understand how things work under the surface.

**Note:** This is a toy project and not meant for production or serious research.

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/dhruvicious/numNet.git
cd numNet
```

## Features

### Core Components

- **Automatic Differentiation Engine**: Custom autograd implementation supporting forward and backward passes for gradient computation
- **Neural Network Layers**: 
  - Linear (fully connected) layers
  - Convolutional layers
  - Pooling layers
  - Flatten layers
  - Dropout layers
  - Sequential container for layer composition

### Optimization

- **Optimizers**: SGD, Adam, and other popular optimization algorithms
- **Loss Functions**: Cross-entropy, Mean Squared Error (MSE), and additional loss functions
- **Activation Functions**: ReLU, Sigmoid, Tanh, GELU, and more

### Initialization

- **Weight Initialization Schemes**: Xavier/Glorot, Kaiming/He, and LeCun initialization methods

## Architecture

numNet follows a modular design pattern similar to modern deep learning frameworks:

- **Tensor Operations**: Built on NumPy arrays with automatic gradient tracking
- **Layer Abstraction**: Clean API for building and composing neural network layers
- **Optimizer Interface**: Standardized optimization procedures for parameter updates

## Limitations

- **Performance**: Pure Python/NumPy implementation without low-level optimizations
- **Hardware**: CPU-only execution (no GPU acceleration)
- **Scale**: Suitable for small to medium-sized experiments
- **Coverage**: Subset of operations compared to production frameworks

## Educational Value

numNet serves as an excellent resource for:

- Understanding backpropagation and automatic differentiation
- Exploring neural network architecture design
- Learning optimization algorithms from first principles
- Prototyping custom layers and operations
- Understanding deep learning fundamentals

## Comparison

| Feature | numNet | PyTorch/TensorFlow |
|---------|--------|-------------------|
| Backend | NumPy (Python) | C++/CUDA |
| Performance | Educational | Production-ready |
| GPU Support | ❌ | ✅ |
| Ecosystem | Non-Existent | Extensive |
| Learning Curve | Gentle | Steeper |

## Acknowledgments

Built just for fun. Inspired by PyTorch's API.

---

*numNet is a toy framework. For actual work, use PyTorch or TensorFlow.*