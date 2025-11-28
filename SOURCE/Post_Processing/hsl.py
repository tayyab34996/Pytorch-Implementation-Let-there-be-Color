import numpy as np
import cv2


def enhance_hsl(img_rgb):
    """Apply the HSL-based enhancement implemented in this module.

    Input: `img_rgb` as an HxWx3 uint8 RGB image.
    Returns: enhanced RGB uint8 image (HxWx3).
    """
    img = img_rgb.copy()
    # convert to float normalized channels
    r = img[:, :, 0].astype(np.float32) / 255.0
    g = img[:, :, 1].astype(np.float32) / 255.0
    b = img[:, :, 2].astype(np.float32) / 255.0

    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn

    # Lightness
    L = (mx + mn) / 2.0

    # Saturation
    S = np.zeros_like(L)
    nonzero = delta != 0
    S[nonzero] = delta[nonzero] / (1 - np.abs(2 * L[nonzero] - 1))

    # Hue
    H = np.zeros_like(L)
    mask = nonzero
    # avoid division by zero where delta==0
    r_m = r[mask]
    g_m = g[mask]
    b_m = b[mask]
    delta_m = delta[mask]
    mx_m = mx[mask]
    h_m = np.zeros_like(r_m)
    # compute hue per channel
    r_eq = mx_m == r_m
    g_eq = mx_m == g_m
    b_eq = mx_m == b_m
    # for r max
    idx = np.where(r_eq)
    if idx[0].size > 0:
        h_m[idx] = (60 * ((g_m[idx] - b_m[idx]) / delta_m[idx]) + 360) % 360
    # for g max
    idx = np.where(g_eq)
    if idx[0].size > 0:
        h_m[idx] = (60 * ((b_m[idx] - r_m[idx]) / delta_m[idx]) + 120) % 360
    # for b max
    idx = np.where(b_eq)
    if idx[0].size > 0:
        h_m[idx] = (60 * ((r_m[idx] - g_m[idx]) / delta_m[idx]) + 240) % 360

    H[mask] = h_m

    # Scale saturation to 0..255 and equalize histogram
    Scaled_saturation = np.uint8(np.round(np.clip(S, 0, 1) * 255))
    hist = np.bincount(Scaled_saturation.flatten(), minlength=256).astype(np.float32)
    cdf = hist.cumsum()
    if cdf[-1] > 0:
        normalized_cdf = np.round((cdf / cdf.max()) * 255).astype(np.uint8)
        equalized_sat = normalized_cdf[Scaled_saturation]
        equalized_sat = equalized_sat.astype(np.float32) / 255.0
    else:
        equalized_sat = Scaled_saturation.astype(np.float32) / 255.0

    # normalize lightness
    min_val = L.min()
    max_val = L.max()
    if max_val > min_val:
        Scaled_lightness = (L - min_val) / (max_val - min_val)
        Scaled_lightness = np.clip(Scaled_lightness, 0.0, 1.0)
    else:
        Scaled_lightness = L

    # Reconstruct RGB from H, equalized S, and scaled L
    Hf = H
    Sf = equalized_sat
    Lf = Scaled_lightness

    C = (1 - np.abs(2 * Lf - 1)) * Sf
    H_div60 = (Hf / 60.0) % 6
    X = C * (1 - np.abs(H_div60 % 2 - 1))
    m = Lf - C / 2.0

    Rp = np.zeros_like(C)
    Gp = np.zeros_like(C)
    Bp = np.zeros_like(C)

    # conditions for H ranges
    cond0 = (Hf >= 0) & (Hf < 60)
    cond1 = (Hf >= 60) & (Hf < 120)
    cond2 = (Hf >= 120) & (Hf < 180)
    cond3 = (Hf >= 180) & (Hf < 240)
    cond4 = (Hf >= 240) & (Hf < 300)
    cond5 = (Hf >= 300) & (Hf < 360)

    Rp[cond0] = C[cond0]
    Gp[cond0] = X[cond0]
    Bp[cond0] = 0

    Rp[cond1] = X[cond1]
    Gp[cond1] = C[cond1]
    Bp[cond1] = 0

    Rp[cond2] = 0
    Gp[cond2] = C[cond2]
    Bp[cond2] = X[cond2]

    Rp[cond3] = 0
    Gp[cond3] = X[cond3]
    Bp[cond3] = C[cond3]

    Rp[cond4] = X[cond4]
    Gp[cond4] = 0
    Bp[cond4] = C[cond4]

    Rp[cond5] = C[cond5]
    Gp[cond5] = 0
    Bp[cond5] = X[cond5]

    R = np.clip((Rp + m) * 255.0, 0, 255).astype(np.uint8)
    G = np.clip((Gp + m) * 255.0, 0, 255).astype(np.uint8)
    B = np.clip((Bp + m) * 255.0, 0, 255).astype(np.uint8)

    final_img = np.dstack((R, G, B))
    return final_img


if __name__ == '__main__':
    # legacy script usage: read low.jpg, run enhance_hsl and show comparison
    import matplotlib.pyplot as plt
    src = cv2.imread('low.jpg')
    if src is not None:
        src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        out = enhance_hsl(src_rgb)
        plt.subplot(1, 2, 1)
        plt.imshow(src_rgb)
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(out)
        plt.title('Enhanced Color Image (HSL-based)')
        plt.axis('off')
        plt.show()
