"""Evaluation utilities: MSE/PSNR, SSIM, LPIPS (or VGG fallback).

Saves per-image CSV to `LOGS/Gland/resnet34_unet_metrics_full.txt` and
prints a summary. Designed to be run from the repository root.
"""
import os
import glob
import cv2
import numpy as np
import torch

LOGS_DIR = os.path.join('LOGS', 'Gland')
RES_DIR = os.path.join('RESULT', 'Gland')
METRICS_FILE = os.path.join(LOGS_DIR, 'resnet34_unet_metrics_full.txt')


def mse(a, b):
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def psnr_from_mse(mse_val, maxv=255.0):
    return float(10 * np.log10((maxv ** 2) / mse_val)) if mse_val > 0 else 100.0


def try_import_ssim():
    try:
        from skimage.metrics import structural_similarity as ssim

        def compute_ssim(a, b):
            # skimage expects images in [H,W,C]. For small images the default
            # window size can exceed the image size and raise an error; compute
            # a safe odd `win_size` <= min(h,w) (and <=7) and pass channel_axis.
            h, w = a.shape[:2]
            win = min(7, h, w)
            if win % 2 == 0:
                win = max(1, win - 1)
            # enforce minimum reasonable window
            if win < 3:
                # fall back to a very small window if image is tiny
                win = 3 if min(h, w) >= 3 else 1
            try:
                # modern skimage uses channel_axis
                s = ssim(a, b, channel_axis=2, win_size=win, data_range=255.0)
            except TypeError:
                # older skimage may use multichannel arg
                s = ssim(a, b, multichannel=True, win_size=win, data_range=255.0)
            except Exception:
                # any other failure -> return None so evaluation continues
                return None
            return float(s)

        return compute_ssim
    except Exception:
        return None


def try_import_lpips(device):
    try:
        import lpips

        loss_fn = lpips.LPIPS(net='vgg').to(device)

        def compute_lpips(a, b):
            # a,b: uint8 HWC in BGR (cv2) -> convert to RGB, float [0,1]
            ar = cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            br = cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            ta = torch.from_numpy(ar).permute(2, 0, 1).unsqueeze(0).to(device)
            tb = torch.from_numpy(br).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                val = loss_fn(ta * 2 - 1, tb * 2 - 1)
            return float(val.cpu().item())

        return compute_lpips
    except Exception:
        return None


def vgg_fallback_lpips(device):
    # fallback to the repo's VGG-based perceptual loss if lpips not installed
    try:
        from losses.perceptual import VGGFeatureExtractor
        import losses.perceptual as perceptual

        extractor = VGGFeatureExtractor(device)

        def compute_vggdist(a, b):
            ar = cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            br = cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            ta = torch.from_numpy(ar).permute(2, 0, 1).unsqueeze(0).to(device)
            tb = torch.from_numpy(br).permute(2, 0, 1).unsqueeze(0).to(device)
            # get features
            with torch.no_grad():
                fa = extractor(ta)
                fb = extractor(tb)
            loss = 0.0
            for x, y in zip(fa, fb):
                loss += torch.nn.functional.mse_loss(x, y).item()
            return float(loss)

        return compute_vggdist
    except Exception:
        return None


def run_evaluation():
    os.makedirs(LOGS_DIR, exist_ok=True)
    files = [f for f in os.listdir(RES_DIR) if f.endswith('reconstructed.jpg')]
    if len(files) == 0:
        print('No reconstructed files found in', RES_DIR)
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ss = try_import_ssim()
    lp = try_import_lpips(device)
    if lp is None:
        lp = vgg_fallback_lpips(device)

    rows = []
    for f in files:
        base = f.replace('reconstructed.jpg', '')
        # find original
        candidates = glob.glob(os.path.join('DATASET', 'Gland', 'test', base + '.*'))
        if not candidates:
            candidates = glob.glob(os.path.join('DATASET', 'Gland', 'train', base + '.*'))
        if not candidates:
            print('Original not found for', f)
            continue
        orig_path = candidates[0]
        orig = cv2.imread(orig_path)
        recon = cv2.imread(os.path.join(RES_DIR, f))
        if orig is None or recon is None:
            print('Failed to read', orig_path, f)
            continue
        if orig.shape != recon.shape:
            recon = cv2.resize(recon, (orig.shape[1], orig.shape[0]))

        m = mse(orig, recon)
        p = psnr_from_mse(m)
        s = ss(orig, recon) if ss is not None else None
        l = lp(orig, recon) if lp is not None else None
        rows.append((f, orig_path, m, p, s, l))

    # write CSV
    with open(METRICS_FILE, 'w') as g:
        g.write('file,orig_path,MSE,PSNR,SSIM,LPIPS_or_VGG\n')
        for r in rows:
            g.write(','.join([str(x) if x is not None else '' for x in r]) + '\n')

    if len(rows) == 0:
        print('No metric pairs computed')
        return

    import numpy as _np
    ms = _np.array([r[2] for r in rows])
    ps = _np.array([r[3] for r in rows])
    ss_vals = _np.array([r[4] for r in rows if r[4] is not None]) if any(r[4] is not None for r in rows) else None
    lp_vals = _np.array([r[5] for r in rows if r[5] is not None]) if any(r[5] is not None for r in rows) else None

    print('Compared', len(rows), 'images')
    print('MSE mean:', ms.mean())
    print('PSNR mean:', ps.mean())
    if ss_vals is not None:
        print('SSIM mean:', ss_vals.mean())
    else:
        print('SSIM not available (skimage missing)')
    if lp_vals is not None:
        print('LPIPS/VGG perceptual mean:', lp_vals.mean())
    else:
        print('LPIPS/VGG not available')
    print('Wrote metrics to', METRICS_FILE)


if __name__ == '__main__':
    run_evaluation()
