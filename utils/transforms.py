import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# 预先定义好 transform 对象
default_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),  # 加这一行，复制灰度为3通道
    transforms.Resize((96, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.405], std=[0.229, 0.224, 0.225])  # 👈 3通道版本
])


def transform_fn(image):
    """这是可被 pickle 的模块级函数，不是 lambda，也不是内部函数"""
    return default_transform(image.astype(np.uint8))


def build_transform(image_size=(36, 60)):
    return transform_fn

