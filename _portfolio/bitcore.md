---
title: "BitCore: Quantization-Aware Ternary Linear Layers for PyTorch"
excerpt: "A PyTorch library providing drop-in ternary linear layers with quantization-aware training and seamless deployment via BitOps, supporting BitNet, TWN, and ParetoQ quantization schemes."
collection: portfolio
---

Training and deploying ternary-quantized neural networks typically requires juggling two very different codepaths: a gradient-aware training path that simulates low-bit arithmetic, and a deployment path that actually runs it. **BitCore** bridges the two with a single `BitLinear` module that works as a drop-in replacement for `nn.Linear`—train with quantization-aware gradients, then flip to optimized inference with one call.

[View the BitCore GitHub Repository](https://github.com/OliverGrainge/BitCore)

## The Problem

Ternary quantization constrains neural network weights to {-1, 0, +1}, enabling massive compression and fast inference via bitwise operations. But standard PyTorch `nn.Linear` layers have no concept of ternary weights. Researchers typically implement custom forward passes with straight-through estimators for training, then manually rewrite inference logic for deployment—an error-prone process that makes it hard to iterate quickly and deploy reliably.

## How BitCore Works

BitCore provides a unified `BitLinear` layer that handles both phases:

1. **Training Mode** (default)
   The layer stores full-precision weights internally but quantizes them on every forward pass using a chosen quantization scheme. Gradients flow through via straight-through estimators, so standard PyTorch optimizers (Adam, SGD, etc.) work out of the box.

2. **Deployment Mode** (`layer.deploy()`)
   A single method call freezes the ternary weights into their packed representation and routes inference through [BitOps](/portfolio/bitops/) for hardware-accelerated matrix multiplication. No code changes to the rest of your model or pipeline.

## Quantization Schemes

BitCore ships with three quantizers, selectable via a string argument:

| Quantizer | Key Idea |
|-----------|----------|
| **BitNet** | Scales weights by their mean absolute value, then rounds to {-1, +1} (default) |
| **TWN** (Ternary Weight Networks) | Adds a zero threshold, producing true ternary {-1, 0, +1} weights with a learned scale |
| **ParetoQ** | Pareto-optimal quantization scheme balancing accuracy and compression |

```python
from bitcore import BitLinear

layer_bitnet  = BitLinear(128, 64, quant_type="bitnet")
layer_twn     = BitLinear(128, 64, quant_type="twn")
layer_paretoq = BitLinear(128, 64, quant_type="paretoq")
```

## Drop-In Model Conversion

Existing models with standard `nn.Linear` layers can be converted in place—no architectural changes required:

```python
from bitcore import BitLinear
import torch.nn as nn

# Convert a single layer
linear = nn.Linear(128, 64)
bitlinear = BitLinear.from_linear(linear, quant_type="bitnet")

# Convert an entire model recursively
def convert_to_bitlinear(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, BitLinear.from_linear(module))
        else:
            convert_to_bitlinear(module)
    return model
```

This is particularly useful for taking a pretrained full-precision model and fine-tuning it with ternary quantization via knowledge distillation—exactly the workflow used in [TAT-VPR](/portfolio/tat/) and [TeTRA-VPR](/portfolio/portfolio-1/).

## Training Example

BitLinear layers are fully compatible with standard PyTorch training loops:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from bitcore import BitLinear

model = nn.Sequential(
    BitLinear(784, 256, quant_type="bitnet"),
    nn.ReLU(),
    BitLinear(256, 10, quant_type="bitnet"),
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

model.train()
for batch in dataloader:
    x, y = batch
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
```

## Deployment

Switching to deployment mode packs the ternary weights and routes computation through BitOps:

```python
model.eval()
for module in model.modules():
    if isinstance(module, BitLinear):
        module.deploy()

with torch.no_grad():
    output = model(x)  # Now uses optimized BitOps kernels
```

## API Reference

### `BitLinear`

```python
BitLinear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    eps: float = 1e-6,
    quant_type: str = "bitnet"
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_features` | `int` | — | Number of input features |
| `out_features` | `int` | — | Number of output features |
| `bias` | `bool` | `True` | Include a bias term |
| `eps` | `float` | `1e-6` | Epsilon for numerical stability |
| `quant_type` | `str` | `"bitnet"` | Quantization scheme: `"bitnet"`, `"twn"`, or `"paretoq"` |

**Methods:**

| Method | Description |
|--------|-------------|
| `forward(x)` | Forward pass with quantization-aware gradients (training) or optimized inference (deployed) |
| `deploy()` | Switch to deployment mode — packs weights and enables BitOps acceleration |
| `from_linear(linear, quant_type)` | Class method to convert an existing `nn.Linear` to `BitLinear` |

## The BitOps Stack

BitCore sits in the middle of a three-layer stack for ternary neural networks:

- **Research models** ([TAT-VPR](/portfolio/tat/), [TeTRA-VPR](/portfolio/portfolio-1/)) define the architectures and training recipes.
- **BitCore** provides the quantization-aware `BitLinear` layer for training and seamless deployment switching.
- **[BitOps](/portfolio/bitops/)** supplies the low-level, hardware-optimized ternary matrix multiplication kernels that BitCore calls in deployment mode.

Together, this stack takes a ternary quantization idea from research prototype to efficient on-device inference with minimal friction.

## Requirements

- Python 3.9+
- PyTorch 2.0.0+
- (Optional) [BitOps](https://github.com/OliverGrainge/BitOps) for deployment-mode acceleration

## License

MIT License
