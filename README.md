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
```

> **Note for Windows users**: Ensure `zlibwapi.dll` is present in your CUDA `bin` directory or system `PATH` (a precompiled binary is included in the root directory).

---

## 🚀 Quick Start

### 1. Run Command-Line Inference

```bash
# Run with GPU acceleration
python main.py --device gpu

# Run with CPU fallback mode
python main.py --device cpu
```

### 2. Run Benchmark Test
```bash
python main_batch.py
```

---

## 💻 Python API Usage

The core inference engine is wrapped in `ocr_lib.OCREngine` for seamless integration into production applications:

```python
import cv2
from ocr_lib.engine import OCREngine

# 1. Initialize Engine (auto handles IOBinding and device allocation)
engine = OCREngine(
    det_model_path='models_onnx/PP_det.onnx',
    rec_model_path='models_onnx/PP_rec.onnx',
    keys_path='models_onnx/ppocrv5_dict.txt',
    device='gpu'  # Options: 'gpu' or 'cpu'
)

# 2. Load input image
image = cv2.imread('test.jpg')

# 3. Perform inference
# Returns: list of tuples -> (bounding_box_coordinates, text_result)
results = engine.predict(image)

for box, text in results:
    print(f"Detected: {text} | Bounding Box: {box}")
```

---

## 🏗️ Architecture & Pipeline

```text
[Input Image (CPU)]
       │ (H2D Memory Copy)
       ▼
[GPU Preprocessing (CuPy / CUDA Kernel Resize & Pad)]
       │
[Detection Inference (ONNX Runtime INT8 with IOBinding)]
       │
[GPU Perspective Transform & Crop (GPU Warp)]
       │
[Recognition Inference (Dynamic Batch INT8)]
       │ (D2H Result Transfer)
       ▼
[DBPostProcess & Output Parsing (Boxes + Text)]
```

### Repository Layout
```text
.
├── ocr_lib/                # Core modular OCR package
│   ├── engine.py           # Unified inference engine (OCREngine)
│   ├── cpu_ops.py          # CPU preprocessing operators (OpenCV)
│   ├── gpu_ops.py          # High-performance GPU kernels (CuPy)
│   └── post_process.py     # Post-processing & box decoding (DBPostProcess)
├── models_onnx/            # ONNX INT8 models & character dictionary
├── main.py                 # Production CLI entry point
├── main_batch.py           # Benchmark script
├── PROJECT_GUIDE_CN.md     # In-depth architectural documentation (CN)
├── README_CN.md            # Detailed user manual (CN)
└── zlibwapi.dll            # CUDA Windows runtime dependency
```

---

## 🗺️ Roadmap

- [ ] Add **TensorRT execution provider** integration for ultra-low latency.
- [ ] Provide automated INT8 post-training quantization (PTQ) calibration scripts.
- [ ] Integrate CI/CD test matrices across mixed CPU and GPU execution providers.
- [ ] Support multilingual dictionaries (English, Japanese, Korean, Latin).

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## 🙏 Acknowledgements

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for the state-of-the-art OCR models and training pipelines.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) for cross-platform inference acceleration.
