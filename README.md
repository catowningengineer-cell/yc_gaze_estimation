# 基于CNN+Transformer的凝视估计混合模型

A hybrid CNN+Transformer model for gaze estimation using PyTorch. It predicts gaze direction from eye images, leveraging CNNs for local feature extraction and Transformers for global context. Trained on MPIIGaze and MPIIFaceGaze with custom preprocessing, angular loss, and flexible training scripts.

这是一个基于 PyTorch 的 CNN+Transformer 混合凝视估计模型。模型从眼部图像中预测视线方向，利用 CNN 进行局部特征提取，结合 Transformer 的全局建模能力。在 MPIIGaze 和 MPIIFaceGaze 数据集上训练，并包含自定义预处理、角度损失和灵活的训练脚本。

## 配置环境

系统：Windows 11 / Ubuntu 20.04，以Ubuntu 20.04为准
显卡：NVIDIA RTX3060/4090
依赖：
```bash
conda create -n gaze-env python=3.10
conda activate gaze-env
conda install pytorch=2.5.1 torchvision=0.20.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install h5py scipy pandas tqdm opencv-python
```
