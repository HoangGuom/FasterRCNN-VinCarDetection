"""
Soft-NMS Implementation - Thay thế Hard-NMS để cải thiện detection
Thay vì xóa hoàn toàn overlapping boxes, ta giảm confidence dần dần
"""

import torch
import numpy as np


def soft_nms_torch(boxes, scores, iou_threshold=0.3, sigma=0.5, keep_top_k=100):
    """
    Soft-NMS implementation using PyTorch
    
    Args:
        boxes: torch.Tensor of shape (N, 4) - Bounding boxes [x1, y1, x2, y2]
        scores: torch.Tensor of shape (N,) - Confidence scores
        iou_threshold: IoU threshold for applying decay
        sigma: Decay parameter (lower = stronger decay for overlapping boxes)
        keep_top_k: Maximum number of boxes to keep
    
    Returns:
        keep_idx: List of indices of kept boxes
        updated_scores: Updated confidence scores
    
    References:
        Improving Object Detection With One Line of Code
        https://arxiv.org/abs/1704.04503
    """
    
    if len(boxes) == 0:
        return [], scores
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores.argsort(descending=True)
    
    keep_idx = []
    updated_scores = scores.clone()
    
    while len(order) > 0:
        i = order[0]
        keep_idx.append(i.item())
        
        if len(order) == 1:
            break
        
        # Các boxes còn lại
        order = order[1:]
        
        # Tính IOU với tất cả các boxes còn lại
        xx1 = torch.maximum(x1[i], x1[order])
        yy1 = torch.maximum(y1[i], y1[order])
        xx2 = torch.minimum(x2[i], x2[order])
        yy2 = torch.minimum(y2[i], y2[order])
        
        w = torch.clamp(xx2 - xx1 + 1.0, min=0)
        h = torch.clamp(yy2 - yy1 + 1.0, min=0)
        inter = w * h
        
        union = areas[i] + areas[order] - inter
        iou = inter.float() / union.float()
        
        # Linear decay: wi = 1 - IoU (nếu IoU > threshold)
        # Exponential decay: wi = exp(-(IoU^2 / sigma))
        weights = torch.ones_like(iou)
        mask = iou > iou_threshold
        weights[mask] = torch.exp(-(iou[mask] ** 2) / sigma)
        
        # Cập nhật scores
        updated_scores[order] = updated_scores[order] * weights
        
        # Sắp xếp lại thu được order mới
        order = order[torch.argsort(updated_scores[order], descending=True)]
    
    # Giữ top-k
    keep_idx = keep_idx[:keep_top_k]
    return keep_idx, updated_scores


def soft_nms_numpy(boxes, scores, iou_threshold=0.3, sigma=0.5, keep_top_k=100):
    """
    Soft-NMS implementation using NumPy (fallback)
    
    Args:
        boxes: np.ndarray of shape (N, 4)
        scores: np.ndarray of shape (N,)
    """
    
    if len(boxes) == 0:
        return [], scores
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = np.argsort(-scores)
    
    keep_idx = []
    updated_scores = scores.copy().astype(np.float32)
    
    while len(order) > 0:
        i = order[0]
        keep_idx.append(i)
        
        if len(order) == 1:
            break
        
        order = order[1:]
        
        # Tính IOU
        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])
        
        w = np.maximum(0, xx2 - xx1 + 1.0)
        h = np.maximum(0, yy2 - yy1 + 1.0)
        inter = w * h
        
        union = areas[i] + areas[order] - inter
        iou = inter.astype(np.float32) / union.astype(np.float32)
        
        # Exponential decay
        weights = np.exp(-(iou ** 2) / sigma)
        weights[iou < iou_threshold] = 1.0
        
        updated_scores[order] = updated_scores[order] * weights
        order = order[np.argsort(-updated_scores[order])]
    
    keep_idx = keep_idx[:keep_top_k]
    return keep_idx, updated_scores
