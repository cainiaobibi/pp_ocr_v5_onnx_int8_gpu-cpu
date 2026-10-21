# PP-OCR v5 ONNX INT8 Inference Engine (GPU & CPU)
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/ONNX%20Runtime-GPU%20%2F%20CPU-orange.svg" alt="ONNX Runtime">
  <img src="https://img.shields.io/badge/Throughput-150%2B%20FPS-red.svg" alt="Performance">
  <img src="https://img.shields.io/badge/CUDA-11.x%20%2F%2012.x-green.svg" alt="CUDA">
</p>
<p align="center">
  <b>High-throughput, production-ready inference engine for PaddleOCR v5 with INT8 quantization, CUDA zero-copy IOBinding, and dynamic batching.</b>
</p>
<p align="center">
  <a href="#features">Features</a> •
  <a href="#benchmark">Benchmark</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#python-api">Python API</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#roadmap">Roadmap</a> •
  <a href="README_CN.md">中文说明</a>
</p>
---
## 🌟 Key Features
- **🚀 Extreme Throughput (150+ FPS)**: Reaches over 150 images per second via ONNX INT8 quantization and full GPU pipeline optimizations.
- **⚡ End-to-End GPU Zero-Copy**: Integrates CuPy CUDA kernels with ONNX Runtime `IOBinding` to eliminate CPU-GPU memory roundtrips.
- **🔄 Fault-Tolerant Auto-Fallback**: Automatically detects CUDA environments; gracefully falls back to CPU execution if GPU initialization or VRAM allocation fails.
- **📦 Modular Production Architecture**: Decoupled clean library design under `ocr_lib` with modular preprocessing, engine inference, and post-processing.
- **🎛️ Dynamic Batching**: Optimized pipeline supporting dynamic batch sizes for detection (`det`) and recognition (`rec`) stages independently.
---
## 📊 Benchmark
Tested on standard industrial OCR verification sets (Image resolution: 960x640):
| Backend | Quantization | Batch Size (Det / Rec) | Throughput (FPS) | Latency / Image |
| :--- | :--- | :--- | :--- | :--- |
| **GPU (CUDA + IOBinding)** | **INT8** | **8 / 64** | **150+ FPS** | **~6.5 ms** |
| GPU (CUDA Standard) | FP32 | 1 / 16 | ~45 FPS | ~22.0 ms |
| CPU (x86_64) | INT8 | 1 / 8 | ~25 FPS | ~40.0 ms |
---
## 🛠️ Requirements & Installation
### Prerequisites
- Python >= 3.8
- CUDA 11.x / 12.x & cuDNN
- Windows or Linux
### Dependencies
Install the required packages via `pip`:
```bash
# For GPU Acceleration
pip install onnxruntime-gpu cupy-cuda12x opencv-python pyclipper numpy
# For CPU-only Execution
pip install onnxruntime opencv-python pyclipper numpy
