import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T


def _lab_to_rgb_tensor(L_norm, ab_norm):
    """Convert Lab (normalized) to RGB tensor in [0,1].

    - L_norm: tensor shape [B,1,H,W], in range [0,1] (originally L/255)
    - ab_norm: tensor shape [B,2,H,W], in range [0,1] (originally ab/255)

    Returns RGB tensor [B,3,H,W] in [0,1]."""
    # Convert to conventional Lab ranges
    # L: 0..100, a/b: -128..127
    L = L_norm * 100.0
    a = ab_norm[:, 0:1, :, :] * 255.0 - 128.0
    b = ab_norm[:, 1:2, :, :] * 255.0 - 128.0

    # f_inv helper
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    def f_inv(t):
        t3 = t ** 3
        mask = t3 > 0.008856
        res = torch.where(mask, t3, (t - 16.0 / 116.0) / 7.787)
        return res

    Xn = 95.047
    Yn = 100.0
    Zn = 108.883

    xr = f_inv(fx)
    yr = f_inv(fy)
    zr = f_inv(fz)

    X = xr * Xn
    Y = yr * Yn
    Z = zr * Zn

    # XYZ to linear RGB (sRGB) matrix (D65)
    X = X / 100.0
    Y = Y / 100.0
    Z = Z / 100.0
    r = X * 3.2406 + Y * -1.5372 + Z * -0.4986
    g = X * -0.9689 + Y * 1.8758 + Z * 0.0415
    b_lin = X * 0.0557 + Y * -0.2040 + Z * 1.0570

    def gamma_correct(channel):
        mask = channel > 0.0031308
        channel = torch.where(mask, 1.055 * torch.pow(channel, 1.0 / 2.4) - 0.055, 12.92 * channel)
        return channel

    rgb = torch.cat([r, g, b_lin], dim=1)
    rgb = gamma_correct(rgb)
    rgb = torch.clamp(rgb, 0.0, 1.0)
    return rgb


class VGGFeatureExtractor(nn.Module):
    def __init__(self, device, pretrained=True, layers=(3, 8, 17)):
        super(VGGFeatureExtractor, self).__init__()
        # layers are indices into vgg.features where ReLU outputs are located
        try:
            vgg = models.vgg19(pretrained=pretrained)
        except Exception:
            vgg = models.vgg19(pretrained=False)
        self.features = vgg.features
        self.layers = layers
        self.to(device)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        # x expected normalized ImageNet tensor [B,3,H,W]
        outputs = []
        current = x
        for i, layer in enumerate(self.features):
            current = layer(current)
            if i in self.layers:
                outputs.append(current)
        return outputs


def perceptual_loss(pred_lab, target_lab, extractor):
    """Compute perceptual loss between predicted and target Lab tensors.

    pred_lab: tuple (L_tensor, ab_tensor) where L [B,1,H,W], ab [B,2,H,W]
    target_lab: same
    extractor: VGGFeatureExtractor on correct device
    """
    Lp, abp = pred_lab
    Lt, abt = target_lab
    # Convert to RGB in [0,1]
    pred_rgb = _lab_to_rgb_tensor(Lp, abp)
    target_rgb = _lab_to_rgb_tensor(Lt, abt)

    # Normalize for ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406], device=pred_rgb.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=pred_rgb.device).view(1, 3, 1, 1)
    pred_in = (pred_rgb - mean) / std
    target_in = (target_rgb - mean) / std

    # Extract features
    with torch.no_grad():
        target_feats = extractor(target_in)
    pred_feats = extractor(pred_in)

    loss = 0.0
    for pf, tf in zip(pred_feats, target_feats):
        loss = loss + F.mse_loss(pf, tf)
    loss = loss / len(pred_feats)
    return loss
