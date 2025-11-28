# -*- coding: utf-8 -*-
"""
Entry point for the PyTorch port. Mirrors the original `main.py` flow:
load data, build model, train, then test.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import data
import resnet34_unet as model
import config
import datetime

if __name__ == "__main__":
    with open(os.path.join(config.LOG_DIR, str(datetime.datetime.now().strftime("%Y%m%d")) + "_" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".txt"), "w") as log:
        log.write(str(datetime.datetime.now()) + "\n")
        log.write("Use Pretrained Weights: " + str(config.USE_PRETRAINED) + "\n")
        log.write("Pretrained Model: " + config.PRETRAINED + "\n")
        # READ DATA
        train_data = data.DATA(config.TRAIN_DIR)
        print("Train Data Loaded")
        # BUILD MODEL
        net = model.MODEL()
        print("Model Initialized")
        net.build()
        print("Model Built")
        # TRAIN MODEL
        net.train_model(train_data, log)
        print("Model Trained")
        # TEST MODEL
        test_data = data.DATA(config.TEST_DIR)
        print("Test Data Loaded")
        net.test_model(test_data, log)
        print("Image Reconstruction Done")
