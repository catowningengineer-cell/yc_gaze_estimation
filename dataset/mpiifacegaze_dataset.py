import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from torchvision import transforms
from PIL import Image


class MPIIFaceGazeDataset(Dataset):
    def __init__(self, h5_path, transform=None, use_headpose=True, participants=None):
        """
        h5_path: 路径，例如 'data/MPIIFaceGaze_processed/MPIIFaceGaze.h5'
        transform: torchvision.transforms（建议包含 Resize, ToTensor, Normalize）
        use_headpose: 是否返回 head pose，与你模型结构匹配
        participants: 如果仅用 p00~p09 训练，p10~p14 测试，则可用来控制子集
        """
        super().__init__()
        self.h5_path = h5_path
        self.transform = transform
        self.use_headpose = use_headpose

        self.samples = []  # 每个样本是 (person_id, index)
        with h5py.File(self.h5_path, 'r') as f:
            keys = list(f.keys())
            if participants is not None:
                keys = [k for k in keys if k in participants]
            for pid in keys:
                N = f[f'{pid}/image'].shape[0]
                for i in range(N):
                    self.samples.append((pid, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, index = self.samples[idx]
        with h5py.File(self.h5_path, 'r') as f:
            image_np = f[f'{pid}/image'][index]  # (H, W, C)
            gaze = f[f'{pid}/gaze'][index]  # (2,)
            pose = f[f'{pid}/pose'][index]  # (2,)
            landmark = f[f'{pid}/landmark'][index]  # (12,)

        image = Image.fromarray(image_np)  # PIL Image

        if self.transform:
            image = self.transform(image)

            # === Landmark resize ===
            original_size = (448, 448)
            target_size = self.transform.transforms[0].size
            if isinstance(target_size, int):
                target_size = (target_size, target_size)

            scale_x = target_size[0] / original_size[0]
            scale_y = target_size[1] / original_size[1]

            landmark[0::2] = landmark[0::2] * scale_x  # x
            landmark[1::2] = landmark[1::2] * scale_y  # y

        result = {
            "image": image,
            "gaze": torch.from_numpy(gaze).float(),
            # "landmark": torch.from_numpy(landmark).float()
        }

        if self.use_headpose:
            result["headpose"] = torch.from_numpy(pose).float()

        return result
