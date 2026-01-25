import os
import time
import numpy as np
import cv2
import onnxruntime as ort

from .post_process import DBPostProcess

class OCREngine:
    def __init__(self, det_model_path, rec_model_path, keys_path, device='cpu'):
        self.device = device.lower()
        self.providers = ['CUDAExecutionProvider'] if self.device == 'gpu' else ['CPUExecutionProvider']
        
        # Session Options
        self.sess_options = ort.SessionOptions()
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Load Models
        print(f"[OCREngine] Loading models on {self.device}...")
        try:
            self.det_sess = ort.InferenceSession(det_model_path, self.sess_options, providers=self.providers)
            self.rec_sess = ort.InferenceSession(rec_model_path, self.sess_options, providers=self.providers)
        except Exception as e:
            if self.device == 'gpu':
                print(f"CRITICAL: GPU Session init failed: {e}. Falling back to CPU.")
                self.device = 'cpu'
                self.providers = ['CPUExecutionProvider']
                self.det_sess = ort.InferenceSession(det_model_path, self.sess_options, providers=self.providers)
                self.rec_sess = ort.InferenceSession(rec_model_path, self.sess_options, providers=self.providers)
            else:
                raise e

        self.det_input_name = self.det_sess.get_inputs()[0].name
        self.rec_input_name = self.rec_sess.get_inputs()[0].name
        
        # Load keys
        self.keys = []
        with open(keys_path, 'r', encoding='utf-8') as f:
            for line in f: self.keys.append(line.strip())
        self.keys = ['blank'] + self.keys + [' ']
        
        self.post_process = DBPostProcess()
        
        # Device Specific Ops
        if self.device == 'gpu':
            import cupy as cp
            from .gpu_ops import ResizeForBatchGPU, normalize_batch_gpu
            self.cp = cp
            self.resize_op = ResizeForBatchGPU(960)
            self.norm_op = normalize_batch_gpu
            # IOBinding support
            self.det_io = self.det_sess.io_binding()
        else:
            from .cpu_ops import ResizeForBatch, normalize_batch
            self.resize_op = ResizeForBatch(960)
            self.norm_op = normalize_batch

    def predict(self, img_orig):
        """
        img_orig: numpy array (H, W, 3)
        Returns: List of (box, text, score)
        """
        if self.device == 'gpu':
            return self._predict_gpu(img_orig)
        else:
            return self._predict_cpu(img_orig)

    def _predict_cpu(self, img):
        from .cpu_ops import get_rotate_crop_image, resize_norm_img_rec
        
        # 1. Det
        resized, ratios = self.resize_op([img])
        norm = self.norm_op(resized)
        preds = self.det_sess.run(None, {self.det_input_name: norm})[0]
        boxes_batch, scores_batch = self.post_process(preds, ratios)
        boxes = boxes_batch[0] if len(boxes_batch) > 0 else []
        
        if len(boxes) == 0: return []
        
        # 2. Rec
        crops = []
        for box in boxes:
            # box is (4, 2)
            crop_pts = box.astype(np.float32)
            crop = get_rotate_crop_image(img, crop_pts)
            crops.append(resize_norm_img_rec(crop))
            
        max_w = max(c.shape[2] for c in crops)
        batch_inputs = np.zeros((len(crops), 3, 48, max_w), dtype=np.float32)
        for i, c in enumerate(crops): batch_inputs[i, :, :, :c.shape[2]] = c
        
        rec_preds = self.rec_sess.run(None, {self.rec_input_name: batch_inputs})[0]
        texts = self._decode(rec_preds)
        
        return list(zip(boxes.tolist(), texts))

    def _predict_gpu(self, img_cpu):
        from .gpu_ops import get_rotate_crop_image_gpu, resize_norm_img_rec_gpu
        cp = self.cp
        
        # Upload
        img_gpu = cp.array(img_cpu)
        
        # 1. Det Pre
        resized_gpu, ratios = self.resize_op([img_gpu])
        norm_gpu = self.norm_op(resized_gpu) # (B, 3, 960, 960)
        
        # 2. Det Infer (IOBinding)
        norm_gpu_contig = cp.ascontiguousarray(norm_gpu)
        out_shape = (1, 1, 960, 960)
        output_gpu = cp.zeros(out_shape, dtype=cp.float32)
        
        self.det_io.bind_input(name=self.det_input_name, device_type='cuda', device_id=0, element_type=np.float32, shape=norm_gpu.shape, buffer_ptr=norm_gpu_contig.data.ptr)
        self.det_io.bind_output(name=self.det_sess.get_outputs()[0].name, device_type='cuda', device_id=0, element_type=np.float32, shape=out_shape, buffer_ptr=output_gpu.data.ptr)
        self.det_sess.run_with_iobinding(self.det_io)
        
        # 3. Post
        preds_cpu = cp.asnumpy(output_gpu)
        boxes_batch, scores_batch = self.post_process(preds_cpu, ratios)
        boxes = boxes_batch[0] if len(boxes_batch) > 0 else []
        if len(boxes) == 0: return []
        
        # 4. Rec Pre (GPU)
        crop_tensors = []
        max_w = 0
        for box in boxes:
            # box is (4, 2)
            crop_pts = box.astype(np.float32)
            cg = get_rotate_crop_image_gpu(img_gpu, crop_pts)
            cn = resize_norm_img_rec_gpu(cg)
            crop_tensors.append(cn)
            max_w = max(max_w, cn.shape[2])
            
        rec_batch = cp.zeros((len(crop_tensors), 3, 48, max_w), dtype=cp.float32)
        for i, c in enumerate(crop_tensors): rec_batch[i, :, :, :c.shape[2]] = c
        
        # 5. Rec Infer (IOBinding)
        rec_io = self.rec_sess.io_binding()
        rec_input_name = self.rec_sess.get_inputs()[0].name
        rec_output_name = self.rec_sess.get_outputs()[0].name
        rec_io.bind_input(name=rec_input_name, device_type='cuda', device_id=0, element_type=np.float32, shape=rec_batch.shape, buffer_ptr=rec_batch.data.ptr)
        rec_io.bind_output(rec_output_name, device_type='cuda', device_id=0)
        self.rec_sess.run_with_iobinding(rec_io)
        rec_preds_cpu = rec_io.get_outputs()[0].numpy()
        
        texts = self._decode(rec_preds_cpu)
        return list(zip(boxes.tolist(), texts))

    def _decode(self, preds):
        idx = preds.argmax(axis=2)
        res = []
        for path in idx:
            t = []
            p = 0
            for i in path:
                if i != 0 and i != p:
                    if i < len(self.keys): t.append(self.keys[i])
                p = i
            res.append("".join(t))
        return res
