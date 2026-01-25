# 工业级高性能 OCR 系统使用说明书

本系统专为高吞吐量工业场景设计，基于 ONNX Runtime (GPU) 和动态批处理 (Dynamic Batching) 技术，可实现 **150+ Images/Second** 的处理速度。

## 1. 环境准备

本系统依赖 `onnxruntime-gpu` 进行加速。请确保您的显卡驱动和 CUDA 环境已正确安装。

### 推荐环境 (Conda)
您可以使用现有的 `HopeScript-Ocr-3.10` 环境：

```bash
conda activate HopeScript-Ocr-3.10
```

### 核心依赖
如果需要手动安装，请确保安装以下库：
- `onnxruntime-gpu` (必须支持 CUDA)
- `opencv-python`
- `numpy`
- `pyclipper` (用于高精度后处理)

## 2. 快速开始

### 运行性能测试 (Benchmark)
系统内置了一个基准测试脚本，会自动生成测试图片并进行全流程推理。

```bash
cd e:\PyProject\OCR
python main_batch.py
```

**预期输出：**
- 控制台将实时打印 FPS (每秒处理张数)。
- 首次运行会自动在 `e:\PyProject\OCR\test_imgs` 生成测试图片。
- 结果图片（画框和识别文字）将保存在 `e:\PyProject\OCR\output` 目录。

### 处理自己的图片
1.  准备一个文件夹，放入需要识别的图片（支持 `.jpg`, `.png`, `.bmp`）。
2.  修改 `main_batch.py` 文件中的 `test_img_dir` 变量，指向您的文件夹路径。

```python
# main_batch.py 第 40 行左右
test_img_dir = "D:/Path/To/Your/Images" 
```
3.  运行脚本：`python main_batch.py`

## 3. 核心配置说明

在 `main_batch.py` 和 `ocr_engine.py` 中有几个关键参数可以调节性能：

| 参数 | 位置 | 说明 | 推荐值 |
| :--- | :--- | :--- | :--- |
| `batch_size` | `main_batch.py:loader` | 检测模型的一次输入批大小。显存越大可设越大。 | `4` - `16` |
| `batch_size_rec` | `ocr_engine.py:process_pipeline` | 识别模型的一次输入批大小。识别模型很小，可设很大。 | `64` - `128` |
| `limit_side_len` | `utils.py:ResizeForBatch` | 输入图像限制长边。越小速度越快，但可能漏检小字。 | `960` (默认) 或 `640` |

## 4. 常见问题排查

**Q: 速度达不到 150 FPS？**
- **检查 GPU**: 确保控制台输出 `Providers: ['CUDAExecutionProvider'...]`。如果是 `CPUExecutionProvider`，请重新安装 `onnxruntime-gpu`。
- **检查 IO**: 如果图片存放在机械硬盘 (HDD) 且分辨率极大（如 4K），读取速度可能是瓶颈。建议放在 SSD。
- **调整 Batch**: 尝试增大 `batch_size` 直到显存占满。

**Q: 报错 `ModuleNotFoundError`?**
- 确保已激活正确的 Conda 环境。
- 缺少 `shapely` 或 `pyclipper` 不会中断运行，但会影响检测框的精细度。建议安装 `pip install pyclipper`。

## 5. 目录结构
```
e:\PyProject\OCR\
├── models_onnx\        # 模型文件 (PP_det_int8.onnx, PP_rec_int8.onnx, dict.txt)
├── output\             # 结果输出目录
├── test_imgs\          # 测试图片目录
├── main_batch.py       # 主程序入口
├── ocr_engine.py       # 推理核心引擎
├── data_loader.py      # 多线程数据加载器
├── utils.py            # 图像预处理与工具函数
└── README_CN.md        # 本说明书
```
