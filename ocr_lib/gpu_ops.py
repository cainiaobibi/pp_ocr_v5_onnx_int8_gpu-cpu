import cupy as cp
import cv2
import numpy as np
import math

# CUDA Kernels
resize_linear_kernel_code = r'''
extern "C" __global__
void resize_linear(const unsigned char* src, unsigned char* dst, 
                   int src_h, int src_w, int src_c, int src_step,
                   int dst_h, int dst_w, int dst_step,
                   float scale_x, float scale_y) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    if (dst_x >= dst_w || dst_y >= dst_h) return;
    float src_x_f = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_y_f = (dst_y + 0.5f) * scale_y - 0.5f;
    int x_low = floorf(src_x_f);
    int y_low = floorf(src_y_f);
    int x_high = x_low + 1;
    int y_high = y_low + 1;
    float delta_x = src_x_f - x_low;
    float delta_y = src_y_f - y_low;
    x_low = max(0, min(x_low, src_w - 1));
    y_low = max(0, min(y_low, src_h - 1));
    x_high = max(0, min(x_high, src_w - 1));
    y_high = max(0, min(y_high, src_h - 1));
    unsigned char v00 = src[y_low * src_step + x_low * src_c + c];
    unsigned char v01 = src[y_low * src_step + x_high * src_c + c];
    unsigned char v10 = src[y_high * src_step + x_low * src_c + c];
    unsigned char v11 = src[y_high * src_step + x_high * src_c + c];
    float val = (1.0f - delta_y) * ((1.0f - delta_x) * v00 + delta_x * v01) + (delta_y) * ((1.0f - delta_x) * v10 + delta_x * v11);
    dst[dst_y * dst_step + dst_x * src_c + c] = (unsigned char)(roundf(val));
}
'''
resize_linear_kernel = cp.RawKernel(resize_linear_kernel_code, 'resize_linear')

warp_perspective_kernel_code = r'''
extern "C" __global__
void warp_perspective(const unsigned char* src, unsigned char* dst,
                      int src_h, int src_w, int src_c, int src_step,
                      int dst_h, int dst_w, int dst_step,
                      const float* M) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    if (dst_x >= dst_w || dst_y >= dst_h) return;
    float x = dst_x;
    float y = dst_y;
    float w_src = M[6] * x + M[7] * y + M[8];
    float x_src = (M[0] * x + M[1] * y + M[2]) / w_src;
    float y_src = (M[3] * x + M[4] * y + M[5]) / w_src;
    int x_low = floorf(x_src);
    int y_low = floorf(y_src);
    int x_high = x_low + 1;
    int y_high = y_low + 1;
    float delta_x = x_src - x_low;
    float delta_y = y_src - y_low;
    if (x_low < 0 || x_high >= src_w || y_low < 0 || y_high >= src_h) {
         if (x_src < 0 || x_src > src_w - 1 || y_src < 0 || y_src > src_h - 1) {
             dst[dst_y * dst_step + dst_x * src_c + c] = 0;
             return;
         }
    }
    int x_low_c = max(0, min(x_low, src_w - 1));
    int y_low_c = max(0, min(y_low, src_h - 1));
    int x_high_c = max(0, min(x_high, src_w - 1));
    int y_high_c = max(0, min(y_high, src_h - 1));
    unsigned char v00 = src[y_low_c * src_step + x_low_c * src_c + c];
    unsigned char v01 = src[y_low_c * src_step + x_high_c * src_c + c];
    unsigned char v10 = src[y_high_c * src_step + x_low_c * src_c + c];
    unsigned char v11 = src[y_high_c * src_step + x_high_c * src_c + c];
    float val = (1.0f - delta_y) * ((1.0f - delta_x) * v00 + delta_x * v01) + (delta_y) * ((1.0f - delta_x) * v10 + delta_x * v11);
    dst[dst_y * dst_step + dst_x * src_c + c] = (unsigned char)(roundf(val));
}
'''
warp_perspective_kernel = cp.RawKernel(warp_perspective_kernel_code, 'warp_perspective')

class ResizeForBatchGPU:
    def __init__(self, limit_side_len=960):
        self.limit_side_len = limit_side_len

    def __call__(self, img_list_gpu):
        batch_size = len(img_list_gpu)
        batch_imgs = cp.zeros((batch_size, self.limit_side_len, self.limit_side_len, 3), dtype=cp.uint8)
        ratios = []
        for i, img in enumerate(img_list_gpu):
            img = cp.ascontiguousarray(img)
            h, w = img.shape[:2]
            ratio = self.limit_side_len / max(h, w)
            resize_h = int(h * ratio)
            resize_w = int(w * ratio)
            resize_h = max(int(round(resize_h / 32) * 32), 32)
            resize_w = max(int(round(resize_w / 32) * 32), 32)
            ratios.append((ratio, resize_h, resize_w))
            resized_gpu = cp.zeros((resize_h, resize_w, 3), dtype=cp.uint8)
            scale_x = float(w) / float(resize_w)
            scale_y = float(h) / float(resize_h)
            block_dim = (32, 32, 1)
            grid_dim = ((resize_w + 31) // 32, (resize_h + 31) // 32, 3)
            resize_linear_kernel(grid_dim, block_dim, (img, resized_gpu, h, w, 3, img.strides[0], resize_h, resize_w, resized_gpu.strides[0], np.float32(scale_x), np.float32(scale_y)))
            batch_imgs[i, :resize_h, :resize_w, :] = resized_gpu
        return batch_imgs, ratios

def normalize_batch_gpu(imgs_gpu):
    imgs_float = imgs_gpu.astype(cp.float32)
    imgs_float *= (1.0 / 255.0)
    mean = cp.array([0.485, 0.456, 0.406], dtype=cp.float32).reshape(1, 1, 1, 3)
    std = cp.array([0.229, 0.224, 0.225], dtype=cp.float32).reshape(1, 1, 1, 3)
    imgs_float -= mean
    imgs_float /= std
    return imgs_float.transpose(0, 3, 1, 2)

def get_rotate_crop_image_gpu(img_gpu, points):
    points = np.array(points).astype(np.float32)
    if isinstance(points, cp.ndarray): points = cp.asnumpy(points)
    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    M_inv = np.linalg.inv(M)
    M_inv_gpu = cp.array(M_inv.flatten().astype(np.float32))
    dst_img = cp.zeros((img_crop_height, img_crop_width, 3), dtype=cp.uint8)
    block_dim = (32, 32, 1)
    grid_dim = ((img_crop_width + 31) // 32, (img_crop_height + 31) // 32, 3)
    warp_perspective_kernel(grid_dim, block_dim, (img_gpu, dst_img, img_gpu.shape[0], img_gpu.shape[1], 3, img_gpu.strides[0], img_crop_height, img_crop_width, dst_img.strides[0], M_inv_gpu))
    if img_crop_height * 1.0 / img_crop_width >= 1.5: dst_img = cp.rot90(dst_img)
    return dst_img

def resize_norm_img_rec_gpu(img_gpu, image_shape=(3, 48, 640)):
    imgC, imgH, imgW = image_shape
    h, w = img_gpu.shape[:2]
    ratio = w / float(h)
    resized_w = min(imgW, int(math.ceil(imgH * ratio)))
    resized_gpu = cp.zeros((imgH, resized_w, 3), dtype=cp.uint8)
    scale_x = float(w) / float(resized_w)
    scale_y = float(h) / float(imgH)
    block_dim = (32, 32, 1)
    grid_dim = ((resized_w + 31) // 32, (imgH + 31) // 32, 3)
    resize_linear_kernel(grid_dim, block_dim, (img_gpu, resized_gpu, h, w, 3, img_gpu.strides[0], imgH, resized_w, resized_gpu.strides[0], np.float32(scale_x), np.float32(scale_y)))
    resized_float = resized_gpu.astype(cp.float32) / 255.0
    resized_float = (resized_float - 0.5) / 0.5
    padding_im = cp.zeros((imgH, imgW, 3), dtype=cp.float32)
    padding_im[:, :resized_w, :] = resized_float
    return padding_im.transpose(2, 0, 1)
