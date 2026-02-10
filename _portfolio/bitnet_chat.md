---
title: "BitNet Chat: Interactive 1.58-bit LLM Demo"
excerpt: "Gradio web app for real-time chat with 1.58-bit BitNet models. 24x speedup and 80% memory reduction on ARM M4 vs PyTorch FP32."
collection: portfolio
header:
  teaser: REPLACE_ME.png
tags:
  - Open Source
  - LLMs
  - Gradio
---

BitNet Chat is an interactive web interface for chatting with extremely compressed large language models in real time. It demonstrates that 1.58-bit quantized models can deliver responsive, coherent conversation while using a fraction of the memory and compute of their full-precision counterparts.

[View the BitNet Chat GitHub Repository](https://github.com/OliverGrainge/BitCore-Demos)

## Overview

The application wraps ternary-quantized BitNet models in a Gradio chat interface, providing a tangible demonstration of the BitCore and BitOps stack in action. Users can interact with compressed LLMs and switch between different backend configurations to compare performance.

## Key Features

- **Real-time streaming responses** with token-by-token generation
- **Backend switching** between different inference engines at runtime
- **24x speedup** over PyTorch FP32 baseline on ARM M4 hardware
- **80% memory reduction** enabling deployment on consumer devices
- **Gradio interface** for zero-setup browser-based interaction

## How It Works

The demo sits at the top of the ternary neural network stack:

1. **BitCore** provides the quantization-aware model layers
2. **BitOps** supplies the hardware-optimized ternary matrix multiplication kernels
3. **BitNet Chat** wraps everything in an accessible chat interface

When a user sends a message, the model generates tokens using packed ternary weights and optimized CUDA or ARM NEON kernels, streaming the response back to the browser in real time.

## Performance

| Metric | FP32 Baseline | BitNet 1.58-bit |
|--------|--------------|-----------------|
| Memory | 100% | 20% |
| Speed (ARM M4) | 1x | 24x |
| Response quality | Baseline | Comparable |

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Gradio
- BitCore & BitOps

## License

MIT License
