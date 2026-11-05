"""
This module provides essential utility functions for the Qwen3-VL video inference pipeline, including configuration management and I/O operations.
It offers helper methods for loading JSON configs, performing atomic file saves, and parsing model predictions from generated text.
The module also includes reproducibility via global seed setting and image processing utilities for resizing video frames.
These shared components are designed to decouple helper logic from the main inference scripts for better modularity.
"""
import json
import os
import random
import numpy as np
import torch
import cv2
from PIL import Image


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(tmp_path, path)

def parse_prediction(text):
    text = text.strip()
    for char in text:
        if char.isdigit():
            return char
    return "N/A"

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resize_video_frame(frame, size=(200,200)):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame_rgb = cv2.resize(frame_rgb, size, interpolation=cv2.INTER_AREA)
    pil_image = Image.fromarray(resized_frame_rgb)
    return pil_image