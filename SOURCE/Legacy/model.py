# -*- coding: utf-8 -*-
"""
PyTorch port of the MODEL class. Implements the same architecture,
loss (MSE) and training/testing loops as the original TensorFlow code.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config
import neural_network
import numpy as np
import cv2


def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)


def reconstruct(batchX, predictedY, filelist):
    for i in range(config.BATCH_SIZE):
        result = np.concatenate((batchX[i], predictedY[i]), axis=2)
        result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
        save_path = os.path.join(config.OUT_DIR, filelist[i][:-4] + "reconstructed.jpg")
        cv2.imwrite(save_path, result)


class MODEL(nn.Module):

    def __init__(self, device=None):
        super(MODEL, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = None
        self.output = None
        # layers will be created in build() to mirror original code flow

    def build(self):
        # Build layers mirroring shapes from the original TF model
        self.low_level_conv1 = neural_network.Convolution_Layer(shape=[3, 3, 1, 64], stddev=0.1, value=0.1)
        self.low_level_conv2 = neural_network.Convolution_Layer(shape=[3, 3, 64, 128], stddev=0.1, value=0.1)
        self.low_level_conv3 = neural_network.Convolution_Layer(shape=[3, 3, 128, 128], stddev=0.1, value=0.1)
        self.low_level_conv4 = neural_network.Convolution_Layer(shape=[3, 3, 128, 256], stddev=0.1, value=0.1)
        self.low_level_conv5 = neural_network.Convolution_Layer(shape=[3, 3, 256, 256], stddev=0.1, value=0.1)
        self.low_level_conv6 = neural_network.Convolution_Layer(shape=[3, 3, 256, 512], stddev=0.1, value=0.1)

        self.mid_level_conv1 = neural_network.Convolution_Layer(shape=[3, 3, 512, 512], stddev=0.1, value=0.1)
        self.mid_level_conv2 = neural_network.Convolution_Layer(shape=[3, 3, 512, 256], stddev=0.1, value=0.1)

        self.global_level_conv1 = neural_network.Convolution_Layer(shape=[3, 3, 512, 512], stddev=0.1, value=0.1)
        self.global_level_conv2 = neural_network.Convolution_Layer(shape=[3, 3, 512, 512], stddev=0.1, value=0.1)
        self.global_level_conv3 = neural_network.Convolution_Layer(shape=[3, 3, 512, 512], stddev=0.1, value=0.1)
        self.global_level_conv4 = neural_network.Convolution_Layer(shape=[3, 3, 512, 512], stddev=0.1, value=0.1)

        # Fully connected equivalents
        # Note: dim will be known at runtime; create placeholders similar to TF by delaying FC weights
        self.global_FC1 = None
        self.global_FC2 = None
        self.global_FC3 = None

        self.fusion_layer = neural_network.Fusion_Layer(shape=[1, 1, 512, 256], stddev=0.1, value=0.1)

        self.colorization_level_conv1 = neural_network.Convolution_Layer(shape=[3, 3, 256, 128], stddev=0.1, value=0.1)
        self.colorization_level_conv2 = neural_network.Convolution_Layer(shape=[3, 3, 128, 64], stddev=0.1, value=0.1)
        self.colorization_level_conv3 = neural_network.Convolution_Layer(shape=[3, 3, 64, 64], stddev=0.1, value=0.1)
        self.colorization_level_conv4 = neural_network.Convolution_Layer(shape=[3, 3, 64, 32], stddev=0.1, value=0.1)

        self.output_layer = neural_network.Output_Layer(shape=[3, 3, 32, 2], stddev=0.1, value=0.1)

    def _build_global_fcs(self, dim):
        # Create FC layers once dim is known (mirrors TF reshape behavior)
        self.global_FC1 = neural_network.FullyConnected_Layer(shape=[dim, 1024], stddev=0.04, value=0.1)
        self.global_FC2 = neural_network.FullyConnected_Layer(shape=[1024, 512], stddev=0.04, value=0.1)
        self.global_FC3 = neural_network.FullyConnected_Layer(shape=[512, 256], stddev=0.04, value=0.1)

    def forward(self, input_batch):
        # input_batch: numpy array [B, H, W, 1], values in [0,1]
        x = torch.from_numpy(input_batch.astype(np.float32)).to(self.device)
        # convert NHWC -> NCHW
        x = x.permute(0, 3, 1, 2)

        h = self.low_level_conv1.feed_forward(x, stride=[1, 2, 2, 1])
        h = self.low_level_conv2.feed_forward(h, stride=[1, 1, 1, 1])
        h = self.low_level_conv3.feed_forward(h, stride=[1, 2, 2, 1])
        h = self.low_level_conv4.feed_forward(h, stride=[1, 1, 1, 1])
        h = self.low_level_conv5.feed_forward(h, stride=[1, 2, 2, 1])
        h = self.low_level_conv6.feed_forward(h, stride=[1, 1, 1, 1])

        h1 = self.mid_level_conv1.feed_forward(h, stride=[1, 1, 1, 1])
        h1 = self.mid_level_conv2.feed_forward(h1, stride=[1, 1, 1, 1])

        h2 = self.global_level_conv1.feed_forward(h, stride=[1, 2, 2, 1])
        h2 = self.global_level_conv2.feed_forward(h2, stride=[1, 1, 1, 1])
        h2 = self.global_level_conv3.feed_forward(h2, stride=[1, 2, 2, 1])
        h2 = self.global_level_conv4.feed_forward(h2, stride=[1, 1, 1, 1])

        # Flatten and pass through FCs
        B = h2.shape[0]
        # ensure tensor is contiguous before view to avoid stride/contiguity errors
        h2_flat = h2.contiguous().view(B, -1)
        dim = h2_flat.shape[1]
        if self.global_FC1 is None:
            self._build_global_fcs(dim)
        h2 = self.global_FC1.feed_forward(h2_flat)
        h2 = self.global_FC2.feed_forward(h2)
        h2 = self.global_FC3.feed_forward(h2)

        # Fusion
        h = self.fusion_layer.feed_forward(h1, h2, stride=[1, 1, 1, 1])

        h = self.colorization_level_conv1.feed_forward(h, stride=[1, 1, 1, 1])
        # upsample to 56x56
        h = F.interpolate(h, size=(56, 56), mode='nearest')
        h = self.colorization_level_conv2.feed_forward(h, stride=[1, 1, 1, 1])
        h = self.colorization_level_conv3.feed_forward(h, stride=[1, 1, 1, 1])
        # upsample to 112x112
        h = F.interpolate(h, size=(112, 112), mode='nearest')
        h = self.colorization_level_conv4.feed_forward(h, stride=[1, 1, 1, 1])
        logits = self.output_layer.feed_forward(h, stride=[1, 1, 1, 1])
        # upsample to 224x224
        out = F.interpolate(logits, size=(224, 224), mode='nearest')

        # convert back to NHWC numpy for reconstruction convenience
        out_np = out.permute(0, 2, 3, 1).cpu().detach().numpy()
        return out_np

    def train_model(self, data, log):
        # optimizer and loss
        params = [p for p in self.parameters() if isinstance(p, nn.Parameter)]
        # Include parameters inside the nested layer objects
        # Scan attributes for Parameter instances
        for name, module in self.__dict__.items():
            if hasattr(module, '__dict__'):
                for v in module.__dict__.values():
                    if isinstance(v, nn.Parameter):
                        params.append(v)
        optimizer = optim.Adam(params, lr=1e-4)
        criterion = nn.MSELoss()

        # move parameters to device
        # (neural_network stores Parameters directly; ensure they are on device)
        for name, val in self.__dict__.items():
            try:
                if isinstance(val, nn.Parameter):
                    val.data = val.data.to(self.device)
            except Exception:
                pass

        if config.USE_PRETRAINED:
            pretrained_path = os.path.join(config.MODEL_DIR, config.PRETRAINED)
            if os.path.exists(pretrained_path):
                checkpoint = torch.load(pretrained_path, map_location=self.device)
                # best-effort load
                try:
                    self.load_state_dict(checkpoint)
                    print('Pretrained weights loaded')
                except Exception:
                    print('Pretrained weights exist but failed to load exact state dict')

        for epoch in range(config.NUM_EPOCHS):
            avg_cost = 0
            total_batches = int(data.size/config.BATCH_SIZE)
            for batch in range(total_batches):
                batchX, batchY, _ = data.generate_batch()
                # forward
                predY = self.forward(batchX)
                # compute loss against labels
                pred_tensor = torch.from_numpy(predY.astype(np.float32)).permute(0,3,1,2).to(self.device)
                label_tensor = torch.from_numpy(batchY.astype(np.float32)).permute(0,3,1,2).to(self.device)
                loss_val = criterion(pred_tensor, label_tensor)
                optimizer.zero_grad()
                # Note: predY is obtained from forward which used parameters; compute gradients indirectly
                # Backpropagate: create a differentiable path by recomputing forward with tensors in grad mode
                # Recompute properly for backprop
                # Recreate input tensor and run the actual parameterized forward
                input_tensor = torch.from_numpy(batchX.astype(np.float32)).permute(0,3,1,2).to(self.device)
                # define a proper forward using the modules' param tensors
                # We'll implement a local differentiable forward using the same operations
                def param_forward(inp):
                    h = self.low_level_conv1.feed_forward(inp, stride=[1,2,2,1])
                    h = self.low_level_conv2.feed_forward(h, stride=[1,1,1,1])
                    h = self.low_level_conv3.feed_forward(h, stride=[1,2,2,1])
                    h = self.low_level_conv4.feed_forward(h, stride=[1,1,1,1])
                    h = self.low_level_conv5.feed_forward(h, stride=[1,2,2,1])
                    h = self.low_level_conv6.feed_forward(h, stride=[1,1,1,1])
                    h1 = self.mid_level_conv1.feed_forward(h, stride=[1,1,1,1])
                    h1 = self.mid_level_conv2.feed_forward(h1, stride=[1,1,1,1])
                    h2 = self.global_level_conv1.feed_forward(h, stride=[1,2,2,1])
                    h2 = self.global_level_conv2.feed_forward(h2, stride=[1,1,1,1])
                    h2 = self.global_level_conv3.feed_forward(h2, stride=[1,2,2,1])
                    h2 = self.global_level_conv4.feed_forward(h2, stride=[1,1,1,1])
                    B = h2.shape[0]
                    # ensure tensor is contiguous before view to avoid stride/contiguity errors
                    h2_flat = h2.contiguous().view(B, -1)
                    dim = h2_flat.shape[1]
                    if self.global_FC1 is None:
                        self._build_global_fcs(dim)
                    h2p = self.global_FC1.feed_forward(h2_flat)
                    h2p = self.global_FC2.feed_forward(h2p)
                    h2p = self.global_FC3.feed_forward(h2p)
                    h = self.fusion_layer.feed_forward(h1, h2p, stride=[1,1,1,1])
                    h = self.colorization_level_conv1.feed_forward(h, stride=[1,1,1,1])
                    h = F.interpolate(h, size=(56,56), mode='nearest')
                    h = self.colorization_level_conv2.feed_forward(h, stride=[1,1,1,1])
                    h = self.colorization_level_conv3.feed_forward(h, stride=[1,1,1,1])
                    h = F.interpolate(h, size=(112,112), mode='nearest')
                    h = self.colorization_level_conv4.feed_forward(h, stride=[1,1,1,1])
                    logits = self.output_layer.feed_forward(h, stride=[1,1,1,1])
                    out = F.interpolate(logits, size=(224,224), mode='nearest')
                    return out

                out_pred = param_forward(input_tensor)
                loss = criterion(out_pred, label_tensor)
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                print("batch:", batch, " loss: ", loss_val)
                avg_cost += loss_val / total_batches

            print("Epoch:", (epoch + 1), "cost =", "{:.5f}".format(avg_cost))
            log.write("Epoch: " + str(epoch + 1) + " Average Cost: " + str(avg_cost) + "\n")

        save_path = os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".pth")
        torch.save(self.state_dict(), save_path)
        print("Model saved in path: %s" % save_path)
        log.write("Model saved in path: " + save_path + "\n")

    def test_model(self, data, log):
        checkpoint_path = os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".pth")
        if os.path.exists(checkpoint_path):
            self.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.eval()
        avg_cost = 0
        total_batch = int(data.size/config.BATCH_SIZE)
        criterion = nn.MSELoss()
        for _ in range(total_batch):
            batchX, batchY, filelist = data.generate_batch()
            predY = self.forward(batchX)
            pred_tensor = torch.from_numpy(predY.astype(np.float32)).permute(0,3,1,2).to(self.device)
            label_tensor = torch.from_numpy(batchY.astype(np.float32)).permute(0,3,1,2).to(self.device)
            loss = criterion(pred_tensor, label_tensor)
            reconstruct(deprocess(batchX), deprocess(predY), filelist)
            avg_cost += loss.item()/total_batch
        print("cost =", "{:.3f}".format(avg_cost))
        log.write("Average Cost: " + str(avg_cost) + "\n")
