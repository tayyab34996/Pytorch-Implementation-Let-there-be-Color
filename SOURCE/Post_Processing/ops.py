import cv2
import numpy as np

# Post-processing operations used by the GUI pipeline.
# All functions accept and return BGR uint8 images.


def clahe(img_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    img = img_bgr.copy()
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        return out
    except Exception:
        return img


def hist_equalize(img_bgr):
    img = img_bgr.copy()
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = cv2.equalizeHist(l)
        lab2 = cv2.merge((l2, a, b))
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        return out
    except Exception:
        return img


def hist_stretch(img_bgr, low_perc=2.0, high_perc=98.0):
    img = img_bgr.copy()
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lo = np.percentile(l, low_perc)
        hi = np.percentile(l, high_perc)
        if hi - lo < 1:
            return img
        l2 = np.clip((l - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
        lab2 = cv2.merge((l2, a, b))
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        return out
    except Exception:
        return img


def hist_shrink(img_bgr, factor=0.8):
    # shrink contrast by moving L towards mid (128)
    img = img_bgr.copy()
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = np.clip(128 + (l.astype(np.float32) - 128.0) * factor, 0, 255).astype(np.uint8)
        lab2 = cv2.merge((l2, a, b))
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        return out
    except Exception:
        return img


def unsharp_mask(img_bgr, amount=1.0, radius=1):
    img = img_bgr.copy()
    try:
        k = max(1, int(radius) * 2 + 1)
        blur = cv2.GaussianBlur(img, (k, k), 0)
        out = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out
    except Exception:
        return img


def lab_chroma_boost(img_bgr, factor=1.2, clip=True):
    img = img_bgr.copy()
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, A, B = cv2.split(lab)
        A2 = A * factor
        B2 = B * factor
        if clip:
            A2 = np.clip(A2, 0, 255)
            B2 = np.clip(B2, 0, 255)
        lab2 = cv2.merge((L, A2, B2)).astype(np.uint8)
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        return out
    except Exception:
        return img


# small utility to apply a named op by params
def apply_op(img_bgr, op_name, params):
    if img_bgr is None:
        return None
    try:
        if op_name == 'CLAHE':
            clip = params.get('clip_limit', 2.0)
            tile = tuple(params.get('tile_grid_size', (8, 8)))
            return clahe(img_bgr, clip_limit=clip, tile_grid_size=tile)
        if op_name == 'Histogram Equalization':
            return hist_equalize(img_bgr)
        if op_name == 'Histogram Stretch':
            low = params.get('low_perc', 2.0)
            high = params.get('high_perc', 98.0)
            return hist_stretch(img_bgr, low_perc=low, high_perc=high)
        if op_name == 'Histogram Shrink':
            factor = params.get('factor', 0.8)
            return hist_shrink(img_bgr, factor=factor)
        if op_name == 'Unsharp Mask':
            amount = params.get('amount', 1.0)
            radius = params.get('radius', 1)
            return unsharp_mask(img_bgr, amount=amount, radius=radius)
        if op_name == 'Lab Chroma Boost':
            factor = params.get('factor', 1.2)
            return lab_chroma_boost(img_bgr, factor=factor)
    except Exception:
        return img_bgr
    return img_bgr
