import numpy as np
import cv2
from pyclipper import PyclipperOffset

class DBPostProcess:
    def __init__(self, thresh=0.3, box_thresh=0.6, unclip_ratio=1.5):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.min_size = 3

    def __call__(self, preds, ratios):
        # preds: (B, 1, H, W)
        pred = preds[:, 0, :, :]
        segmentation = pred > self.thresh
        
        boxes_batch = []
        scores_batch = []
        
        for i in range(len(segmentation)):
            mask = segmentation[i].astype(np.uint8) * 255
            ratio_h, ratio_w = ratios[i][1:] # resize_h, resize_w
            # Real ratio
            # This logic needs to match the ResizeForBatch logic
            # ratios[i] = (ratio, resize_h, resize_w)
            
            num_h, num_w = pred.shape[1:]
            boxes, scores = self.boxes_from_bitmap(pred[i], mask, ratios[i][2], ratios[i][1])
            
            # Rescale boxes to original image
            # original_max_side / limit_side_len ?
            # actually we used ratio = limit / max(h,w)
            # so src = dst / ratio
            ratio = ratios[i][0]
            boxes = boxes / ratio
            
            boxes_batch.append(np.array(boxes))
            scores_batch.append(scores)
            
        return boxes_batch, scores_batch

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        bitmap = _bitmap[:dest_height, :dest_width]
        pred = pred[:dest_height, :dest_width]
        
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        scores = []
        
        for contour in contours:
            points, side = self.get_mini_boxes(contour)
            if side < self.min_size:
                continue
            
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if score < self.box_thresh:
                continue
                
            box = self.unclip(points).reshape(-1, 1, 2)
            box, side = self.get_mini_boxes(box)
            if side < self.min_size + 2:
                continue
            box = np.array(box)
            
            boxes.append(box)
            scores.append(score)
            
        return np.array(boxes), scores

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _points):
        h, w = bitmap.shape[:2]
        points = _points.copy()
        xmin = np.clip(np.floor(points[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(points[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(points[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(points[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        points[:, 0] -= xmin
        points[:, 1] -= ymin
        cv2.fillPoly(mask, points.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def unclip(self, box):
        poly = box
        distance = self.unclip_ratio * cv2.contourArea(poly) / cv2.arcLength(poly, True)
        offset = PyclipperOffset()
        import pyclipper
        offset.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded
