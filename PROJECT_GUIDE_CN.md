# OCR 生产工程封装技术文档 (Project Guide)

本项目已完成从实验脚本到生产级工程的封装。核心逻辑已迁移至 `ocr_lib` 模块。

## 1. 项目目录结构

```text
OCR/
├── main.py                 # 生产环境入口脚本
├── ocr_lib/                # 核心 OCR 库
│   ├── __init__.py         # 包初始化
│   ├── engine.py           # 统一推理引擎接口 (OCREngine)
│   ├── cpu_ops.py          # CPU 预处理算子 (OpenCV)
│   ├── gpu_ops.py          # GPU 预处理算子 (CuPy + CUDA Kernel)
│   └── post_process.py     # 共享后处理逻辑 (DBPostProcess)
├── models_onnx/            # ONNX 模型及字典文件
└── output/                 # 结果导出目录
```

## 2. 核心 API 使用说明

通过 `ocr_lib.OCREngine` 类，您可以轻松切换计算后端：

```python
from ocr_lib.engine import OCREngine

# 初始化引擎
# device: 'gpu' 或 'cpu'
# 自动处理硬件分配、显存零拷贝绑定 (IOBinding)
engine = OCREngine(
    det_model_path='models_onnx/PP_det.onnx',
    rec_model_path='models_onnx/PP_rec.onnx',
    keys_path='models_onnx/ppocrv5_dict.txt',
    device='gpu'
)

# 执行预测
# 输入：numpy 格式图像 (HWC, BGR)
# 返回：list 嵌套 (box, text)
results = engine.predict(image)

for box, text in results:
    print(f"找到文字: {text}, 位置: {box}")
```

## 3. 技术优化点

### 3.1 显存内流水线 (GPU Mode)
在 GPU 模式下，项目实现了 **“图像进，文字出”** 的全显存流水线：
*   **输入上传**：图片一次性由 CPU 拷贝至 GPU。
*   **预处理**：使用自定义 CUDA Kernel 进行 Resize 和 Pad，结果直接留在显存。
*   **推理 (IOBinding)**：利用 ONNXRuntime 的 IOBinding 技术，将推理引擎的输入输出指针直接指向显存地址，绕过 CPU 内存周转。
*   **裁剪 (GPU Warp)**：利用 GPU 加速透视变换处理识别框。

### 3.2 自动降级机制
如果系统初始化 GPU 环境（如驱动不匹配或显存不足）失败，`OCREngine` 会自动捕获异常并降级到 **CPU 模式** 运行，确保业务不中断。

## 4. 运行说明

**使用 GPU 加速运行：**
```bash
python main.py --device gpu
```

**使用 CPU 兼容模式运行：**
```bash
python main.py --device cpu
```

## 5. 依赖项
*   `onnxruntime-gpu` / `onnxruntime`
*   `cupy` (仅 GPU 模式需要)
*   `opencv-python`
*   `pyclipper`
*   `numpy`
*   `zlibwapi.dll`：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin` / `C:\Users\catmi\.conda\envs\HopeScript-Ocr-3.10\Library\bin` 