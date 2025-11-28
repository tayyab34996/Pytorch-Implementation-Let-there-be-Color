# -*- coding: utf-8 -*-
"""
ResNet34 encoder + simple UNet-like decoder colorization model.

Provides a `MODEL` class with `build()`, `train_model()`, `test_model()` and
`forward()` methods so it can be used as a drop-in replacement for the
original `model.MODEL` in `SOURCE/main.py`.

This implementation uses a torchvision ResNet34 backbone (optionally
pretrained) and a lightweight decoder that upsamples features and predicts
2-channel `ab` output normalized to [0,1] (same convention as the current
codebase where lab channels are divided by 255).
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import config
from losses.perceptual import VGGFeatureExtractor, perceptual_loss
import torch.cuda.amp as amp


def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)


def reconstruct(batchX, predictedY, filelist):
    # batchX: numpy [B,H,W,1] values in [0,1]
    # predictedY: numpy [B,H,W,2] values in [0,1]
    for i in range(predictedY.shape[0]):
        result = np.concatenate((batchX[i], predictedY[i]), axis=2)
        result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
        save_path = os.path.join(config.OUT_DIR, filelist[i][:-4] + "reconstructed.jpg")
        cv2.imwrite(save_path, result)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # squeeze-and-excitation block will be attached after convs
        self.se = SEBlock(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block (channel attention).

    Lightweight implementation using global average pooling and two FC layers.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        hidden = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class RefinementBlock(nn.Module):
    """Small residual refinement block: two convs with BN + ReLU and a residual add."""
    def __init__(self, channels):
        super(RefinementBlock, self).__init__()
        hidden = max(channels // 2, 16)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class MODEL(nn.Module):
    def __init__(self, device=None):
        super(MODEL, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = None

    def build(self):
        # Load resnet34 backbone
        try:
            resnet = models.resnet34(pretrained=True)
        except Exception:
            resnet = models.resnet34(pretrained=False)

        # Use layers as encoder
        self.enc_conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc_pool = resnet.maxpool
        self.enc_layer1 = resnet.layer1  # 64
        self.enc_layer2 = resnet.layer2  # 128
        self.enc_layer3 = resnet.layer3  # 256
        self.enc_layer4 = resnet.layer4  # 512

        # Decoder: progressively upsample and combine
        self.dec4 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256 + 256, 128)
        self.dec2 = DecoderBlock(128 + 128, 64)
        self.dec1 = DecoderBlock(64 + 64, 64)
        # Refinement block placed before final conv to clean up output
        self.refine = RefinementBlock(64)
        self.final_conv = nn.Conv2d(64, 2, kernel_size=1)

        # move to device
        self.to(self.device)

    def forward_tensor(self, x_tensor):
        """Differentiable forward that accepts a torch tensor in NCHW format
        and returns a tensor output (NCHW) so autograd can track gradients.
        Expect input shape [B, 3, H, W] (3 channels)."""
        c1 = self.enc_conv1(x_tensor)  # [B,64,H/2,W/2]
        p = self.enc_pool(c1)
        c2 = self.enc_layer1(p)  # [B,64,H/4,W/4]
        c3 = self.enc_layer2(c2)  # [B,128,H/8,W/8]
        c4 = self.enc_layer3(c3)  # [B,256,H/16,W/16]
        c5 = self.enc_layer4(c4)  # [B,512,H/32,W/32]

        d4 = F.interpolate(self.dec4(c5), scale_factor=2, mode='bilinear', align_corners=False)
        d3 = torch.cat([d4, c4], dim=1)
        d3 = F.interpolate(self.dec3(d3), scale_factor=2, mode='bilinear', align_corners=False)
        d2 = torch.cat([d3, c3], dim=1)
        d2 = F.interpolate(self.dec2(d2), scale_factor=2, mode='bilinear', align_corners=False)
        d1 = torch.cat([d2, c2], dim=1)
        d1 = F.interpolate(self.dec1(d1), scale_factor=2, mode='bilinear', align_corners=False)
        # refinement before final projection
        d1 = self.refine(d1)

        out = self.final_conv(d1)
        out = F.interpolate(out, size=(config.IMAGE_SIZE, config.IMAGE_SIZE), mode='bilinear', align_corners=False)
        # apply sigmoid to map to [0,1]
        out = torch.sigmoid(out)
        return out

    def forward(self, input_batch):
        """Flexible forward: accepts either a NumPy NHWC input (as before) and
        returns a NumPy NHWC output, or accepts a torch tensor (NCHW, 3-ch)
        and returns a torch tensor (NCHW).
        """
        if isinstance(input_batch, np.ndarray):
            x = torch.from_numpy(input_batch.astype(np.float32)).to(self.device)
            x = x.permute(0, 3, 1, 2)
            x = x.repeat(1, 3, 1, 1)
            out = self.forward_tensor(x)
            out_np = out.permute(0, 2, 3, 1).cpu().detach().numpy()
            return out_np
        elif torch.is_tensor(input_batch):
            return self.forward_tensor(input_batch)
        else:
            raise TypeError('Unsupported input type for forward')

    def train_model(self, data, log, progress_callback=None):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        use_perc = config.USE_PERCEPTUAL and torch.cuda.is_available()
        if config.USE_PERCEPTUAL and not torch.cuda.is_available():
            print('Perceptual loss requested but CUDA not available — disabling perceptual term for this run')
            use_perc = False

        extractor = None
        if use_perc:
            extractor = VGGFeatureExtractor(self.device)

        use_amp = config.USE_AMP and torch.cuda.is_available()
        scaler = amp.GradScaler() if use_amp else None

        # Resume support: if a checkpoint path is provided in config.RESUME_FROM
        start_epoch = 0
        resume_path = os.environ.get('RESUME_FROM', '') or config.RESUME_FROM
        if resume_path:
            if os.path.exists(resume_path):
                print('Resuming from checkpoint:', resume_path)
                ckpt = torch.load(resume_path, map_location=self.device)
                # support both state_dict-only and dict checkpoints
                if isinstance(ckpt, dict) and 'model_state' in ckpt:
                    self.load_state_dict(ckpt['model_state'])
                    if 'optimizer_state' in ckpt and optimizer is not None:
                        try:
                            optimizer.load_state_dict(ckpt['optimizer_state'])
                        except Exception:
                            print('Warning: optimizer state could not be fully loaded')
                    if 'scaler_state' in ckpt and scaler is not None:
                        try:
                            scaler.load_state_dict(ckpt['scaler_state'])
                        except Exception:
                            print('Warning: scaler state could not be loaded')
                    start_epoch = int(ckpt.get('epoch', 0))
                else:
                    # legacy: assume file contains only the model state_dict
                    try:
                        self.load_state_dict(ckpt)
                    except Exception:
                        print('Failed to load checkpoint as state_dict')
            else:
                print('Requested resume checkpoint not found:', resume_path)

        total_batches = int(data.size / config.BATCH_SIZE)
        total_steps = max(1, config.NUM_EPOCHS * total_batches)
        for epoch in range(start_epoch, config.NUM_EPOCHS):
            avg_cost = 0
            self.train()
            for batch in range(total_batches):
                batchX, batchY, _ = data.generate_batch()
                # input_l: [B,1,H,W]
                input_l = torch.from_numpy(batchX.astype(np.float32)).permute(0, 3, 1, 2).to(self.device)
                input_tensor = input_l.repeat(1, 3, 1, 1)
                label_tensor = torch.from_numpy(batchY.astype(np.float32)).permute(0, 3, 1, 2).to(self.device)

                optimizer.zero_grad()
                if use_amp:
                    with amp.autocast():
                        pred_tensor = self.forward_tensor(input_tensor)
                        loss_pixel = criterion(pred_tensor, label_tensor)
                        loss = loss_pixel
                        if use_perc and epoch + 1 >= config.PERCEPTUAL_START_EPOCH:
                            # prepare Lab tuples: L in [B,1,H,W], ab in [B,2,H,W]
                            pred_lab = (input_l, pred_tensor)
                            target_lab = (input_l, label_tensor)
                            perc = perceptual_loss(pred_lab, target_lab, extractor)
                            loss = loss + config.PERCEPTUAL_WEIGHT * perc
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred_tensor = self.forward_tensor(input_tensor)
                    loss_pixel = criterion(pred_tensor, label_tensor)
                    loss = loss_pixel
                    if use_perc and epoch + 1 >= config.PERCEPTUAL_START_EPOCH:
                        pred_lab = (input_l, pred_tensor)
                        target_lab = (input_l, label_tensor)
                        perc = perceptual_loss(pred_lab, target_lab, extractor)
                        loss = loss + config.PERCEPTUAL_WEIGHT * perc
                    loss.backward()
                    optimizer.step()

                loss_val = loss_pixel.item()
                print("batch:", batch, " loss: ", loss_val)
                avg_cost += loss_val / total_batches
                # progress callback: (epoch, batch, total_batches, percent, loss)
                if progress_callback is not None:
                    try:
                        step_index = epoch * total_batches + batch
                        percent = int(100.0 * (step_index + 1) / max(1, total_steps))
                        progress_callback(epoch + 1, batch + 1, total_batches, percent, loss_val)
                    except Exception:
                        pass

            print("Epoch:", (epoch + 1), "cost =", "{:.5f}".format(avg_cost))
            log.write("Epoch: " + str(epoch + 1) + " Average Cost: " + str(avg_cost) + "\n")
            # save checkpoint at epoch (1-indexed)
            ep_num = epoch + 1
            if config.CHECKPOINT_FREQ and (ep_num % config.CHECKPOINT_FREQ == 0):
                ckpt = {
                    'epoch': ep_num,
                    'model_state': self.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                }
                if scaler is not None:
                    ckpt['scaler_state'] = scaler.state_dict()
                save_path = os.path.join(config.MODEL_DIR, f"model{config.BATCH_SIZE}_{ep_num}.pth")
                torch.save(ckpt, save_path)
                print("Checkpoint saved:", save_path)
                log.write("Checkpoint saved: " + save_path + "\n")

        # final save (also store as final epoch name)
        final_ckpt = {
            'epoch': config.NUM_EPOCHS,
            'model_state': self.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        if scaler is not None:
            final_ckpt['scaler_state'] = scaler.state_dict()
        final_path = os.path.join(config.MODEL_DIR, f"model{config.BATCH_SIZE}_{config.NUM_EPOCHS}.pth")
        torch.save(final_ckpt, final_path)
        print("Model saved in path: %s" % final_path)
        log.write("Model saved in path: " + final_path + "\n")

    def test_model(self, data, log):
        # prefer explicit resume checkpoint if provided, otherwise use expected final checkpoint
        resume_path = os.environ.get('RESUME_FROM', '') or config.RESUME_FROM
        if resume_path and os.path.exists(resume_path):
            ckpt = torch.load(resume_path, map_location=self.device)
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                self.load_state_dict(ckpt['model_state'])
            else:
                self.load_state_dict(ckpt)
        else:
            checkpoint_path = os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".pth")
            if os.path.exists(checkpoint_path):
                ck = torch.load(checkpoint_path, map_location=self.device)
                if isinstance(ck, dict) and 'model_state' in ck:
                    self.load_state_dict(ck['model_state'])
                else:
                    self.load_state_dict(ck)
        self.eval()
        avg_cost = 0
        total_batch = int(data.size / config.BATCH_SIZE)
        criterion = nn.MSELoss()
        for _ in range(total_batch):
            batchX, batchY, filelist = data.generate_batch()
            predY = self.forward(batchX)
            pred_tensor = torch.from_numpy(predY.astype(np.float32)).permute(0, 3, 1, 2).to(self.device)
            label_tensor = torch.from_numpy(batchY.astype(np.float32)).permute(0, 3, 1, 2).to(self.device)
            loss = criterion(pred_tensor, label_tensor)
            reconstruct(deprocess(batchX), deprocess(predY), filelist)
            avg_cost += loss.item() / total_batch
        print("cost =", "{:.3f}".format(avg_cost))
        log.write("Average Cost: " + str(avg_cost) + "\n")
