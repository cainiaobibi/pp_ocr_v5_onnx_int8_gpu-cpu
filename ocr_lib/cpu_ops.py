import cv2
import numpy as np
import math

class ResizeForBatch:
    def __init__(self, limit_side_len=960):
        self.limit_side_len = limit_side_len

    def __call__(self, img_list):
        batch_size = len(img_list)
        batch_imgs = np.zeros((batch_size, self.limit_side_len, self.limit_side_len, 3), dtype=np.uint8)
        ratios = []
        for i, img in enumerate(img_list):
            h, w = img.shape[:2]
            ratio = self.limit_side_len / max(h, w)
            resize_h = int(h * ratio)
            resize_w = int(w * ratio)
            
            # Ensure multiple of 32
            resize_h = max(int(round(resize_h / 32) * 32), 32)
            resize_w = max(int(round(resize_w / 32) * 32), 32)
            
            resized = cv2.resize(img, (resize_w, resize_h))
            batch_imgs[i, :resize_h, :resize_w, :] = resized
            ratios.append((ratio, resize_h, resize_w))
            
        return batch_imgs, ratios

def normalize_batch(imgs, scale=1.0/255.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    imgs = imgs.astype('float32') * scale
    mean = np.array(mean).reshape(1, 1, 1, 3).astype('float32')
    std = np.array(std).reshape(1, 1, 1, 3).astype('float32')
    imgs -= mean
    imgs /= std
    return imgs.transpose(0, 3, 1, 2)

def get_rotate_crop_image(img, points):
    points = np.array(points).astype(np.float32)
    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if img_crop_height * 1.0 / img_crop_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

def resize_norm_img_rec(img, image_shape=(3, 48, 320)):
    imgC, imgH, imgW = image_shape
    h, w = img.shape[:2]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    resized_image = resized_image / 255.0
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgH, imgW, 3), dtype=np.float32)
    padding_im[:, 0:resized_w, :] = resized_image
    return padding_im.transpose(2, 0, 1)
