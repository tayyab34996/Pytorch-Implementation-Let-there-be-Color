"""Simple script to colorize a single grayscale image using the trained model.

Usage:
  conda activate colorize-fresh
  python SOURCE/colorize_image.py input_gray.jpg output_color.jpg [checkpoint_path]

If `checkpoint_path` is omitted the script will try `SOURCE/config.RESUME_FROM` or
`MODEL/model<BATCH>_<NUM_EPOCHS>.pth`.
"""
import sys
import os
import cv2
import numpy as np
import torch
import config
from resnet34_unet import MODEL


def load_checkpoint(model, path, device):
    if not path:
        path = os.environ.get('RESUME_FROM', '') or config.RESUME_FROM
    if not path:
        # default final model
        path = os.path.join(config.MODEL_DIR, f"model{config.BATCH_SIZE}_{config.NUM_EPOCHS}.pth")
    if not os.path.exists(path):
        print('Checkpoint not found:', path)
        return False
    ck = torch.load(path, map_location=device)
    if isinstance(ck, dict) and 'model_state' in ck:
        model.load_state_dict(ck['model_state'])
    else:
        model.load_state_dict(ck)
    print('Loaded checkpoint:', path)
    return True


def _pad_to_square(img, target_size, border_value=0):
    h, w = img.shape[:2]
    if h == target_size and w == target_size:
        return img, (0, 0, h, w)
    # scale preserving aspect ratio so longer side == target_size
    scale = target_size / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    resized = cv2.resize(img, (new_w, new_h))
    # pad to target_size
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
    return padded, (pad_top, pad_left, new_h, new_w)


def _unpad_and_resize(out_bgr, orig_shape, pad_info):
    pad_top, pad_left, new_h, new_w = pad_info
    # crop to the resized area then scale back to original shape
    cropped = out_bgr[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    return cv2.resize(cropped, (orig_shape[1], orig_shape[0]))


def colorize_image(in_path, out_path, checkpoint=None, batch=False, method='model'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MODEL(device=device)
    model.build()
    ok = load_checkpoint(model, checkpoint, device)
    if not ok:
        print('Continuing with uninitialized weights.')

    def process_file(src, dst):
        # load grayscale (or single-channel) image
        img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        if img is None:
            print('Failed to read', src)
            return
        # if 3-channel, decide behavior based on requested method
        if len(img.shape) == 3 and img.shape[2] == 3:
            if method == 'reconstruct':
                # use the L channel from Lab (preserves perceptual luminance)
                try:
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
                    gray = lab[..., 0]
                except Exception:
                    # fallback to simple grayscale if conversion fails
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                # default / 'grayscale' behavior: convert to grayscale image
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Save grayscale copy next to the color output for GUI / later use
        try:
            dst_dir = os.path.dirname(dst) or '.'
            base, ext = os.path.splitext(os.path.basename(dst))
            gray_name = f"{base}_gray{ext}"
            gray_path = os.path.join(dst_dir, gray_name)
            cv2.imwrite(gray_path, gray)
        except Exception:
            pass

        H = config.IMAGE_SIZE
        padded, pad_info = _pad_to_square(gray, H)
        l_norm = (padded.astype(np.float32) / 255.0).reshape(1, H, H, 1)

        out_ab = model.forward(l_norm)  # NHWC
        out_ab = out_ab[0]
        L = l_norm[0]
        lab = np.concatenate([L, out_ab], axis=2)
        lab = (lab * 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        final = _unpad_and_resize(bgr, gray.shape, pad_info)
        cv2.imwrite(dst, final)
        print('Wrote colorized image to', dst)

    if batch and os.path.isdir(in_path):
        os.makedirs(out_path, exist_ok=True)
        files = [f for f in os.listdir(in_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
        for f in files:
            name, ext = os.path.splitext(f)
            dst_name = f"{name}_color{ext}"
            process_file(os.path.join(in_path, f), os.path.join(out_path, dst_name))
    else:
        process_file(in_path, out_path)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python SOURCE/colorize_image.py input_gray.jpg output_color.jpg [checkpoint_path]')
        sys.exit(1)
    in_p = sys.argv[1]
    out_p = sys.argv[2]
    ck = sys.argv[3] if len(sys.argv) > 3 else None
    colorize_image(in_p, out_p, ck)
