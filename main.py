import cv2
import argparse
import time
import os
from ocr_lib.engine import OCREngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='img.png')
    parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'])
    parser.add_argument('--det', type=str, default='models_onnx/PP_det.onnx')
    parser.add_argument('--rec', type=str, default='models_onnx/PP_rec.onnx')
    parser.add_argument('--keys', type=str, default='models_onnx/ppocrv5_dict.txt')
    args = parser.parse_args()

    # Load DLL fix if on Windows for GPU
    if args.device == 'gpu':
        cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
        if os.path.exists(cuda_path):
            os.environ['PATH'] = cuda_path + os.pathsep + os.environ['PATH']

    engine = OCREngine(args.det, args.rec, args.keys, device=args.device)
    
    img = cv2.imread(args.img)
    if img is None:
        print(f"Error loading {args.img}")
        return

    print(f"Running prediction on {args.device}...")
    t0 = time.time()
    results = engine.predict(img)
    fps = 1.0 / (time.time() - t0)
    
    print(f"\nResults (FPS: {fps:.2f}):")
    for box, text in results:
        print(f"{text}：{box}")

    # Save visualization
    for box, text in results:
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
    
    out_name = f"output_prod_{args.device}.jpg"
    cv2.imwrite(out_name, img)
    print(f"\nVisual result saved to {out_name}")

if __name__ == "__main__":
    import numpy as np
    main()
