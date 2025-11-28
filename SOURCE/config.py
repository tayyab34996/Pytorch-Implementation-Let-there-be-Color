# -*- coding: utf-8 -*-
"""
PyTorch port config. Mirrors original project config and points
to the original repo directories so datasets/results are reused.
"""
import os

# DIRECTORY INFORMATION
# Use this repository root so DATASET/RESULT/MODEL/LOGS
# are relative to the current project workspace.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Default dataset in this workspace
DATASET = "Gland"
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET', DATASET)
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT', DATASET)
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL')
LOG_DIR = os.path.join(ROOT_DIR, 'LOGS', DATASET)

TRAIN_DIR = "train"
TEST_DIR = "test"

# DATA INFORMATION
IMAGE_SIZE = 224
# Allow overriding batch size via environment variable for quick tests/smoke runs
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 1))

# RANDOM NUMBER GENERATOR INFORMATION
SEED = 128

# TRAINING INFORMATION
USE_PRETRAINED = False
PRETRAINED = "Dogsmodel1_100"
# Allow overriding number of epochs via environment variable for smoke runs
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 100))
# Perceptual loss options
USE_PERCEPTUAL = bool(int(os.environ.get('USE_PERCEPTUAL', 0)))
PERCEPTUAL_WEIGHT = float(os.environ.get('PERCEPTUAL_WEIGHT', 0.01))
PERCEPTUAL_START_EPOCH = int(os.environ.get('PERCEPTUAL_START_EPOCH', 1))
# Mixed precision autocast
USE_AMP = bool(int(os.environ.get('USE_AMP', 1)))
# Checkpoint / resume options
# Path to a checkpoint file to resume from (env or empty)
RESUME_FROM = os.environ.get('RESUME_FROM', '')
# How often (in epochs) to save intermediate checkpoints. 1 = every epoch.
CHECKPOINT_FREQ = int(os.environ.get('CHECKPOINT_FREQ', 5))
