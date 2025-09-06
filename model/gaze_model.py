import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class GazeTransformer(nn.Module):
    def __init__(self, feature_dim=256, transformer_layers=6, transformer_heads=4, use_headpose=True):
        super().__init__()

        self.use_headpose = use_headpose

        # CNN Backbone (ResNet18, 去掉 avgpool 和 fc)
        base_cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cnn_backbone = nn.Sequential(
            base_cnn.conv1,
            base_cnn.bn1,
            base_cnn.relu,
            base_cnn.maxpool,
            base_cnn.layer1,
            base_cnn.layer2,
            base_cnn.layer3  # 停在 layer3，输出 stride=16，通道为 256
        )

        # 1x1 conv 降维
        self.conv_proj = nn.Conv2d(256, feature_dim, kernel_size=1)

        # 可学习的位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, 100, feature_dim))  # 最多 100 patch

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=transformer_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Gaze 方向回归器
        head_input_dim = feature_dim + (2 if use_headpose else 0)
        self.regressor = nn.Sequential(
            nn.Linear(head_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 2)  # [pitch, yaw]
        )

    def forward(self, x, head_pose=None):
        # CNN 特征提取
        x = self.cnn_backbone(x)             # [B, 512, H', W']
        x = self.conv_proj(x)                # [B, feature_dim, H', W']
        B, C, H, W = x.shape

        # flatten 为 patch 序列
        x = x.flatten(2).transpose(1, 2)     # [B, N, C], N=H*W

        # 添加位置编码（裁剪或扩展）
        if x.size(1) > self.pos_embed.size(1):
            raise ValueError("Position embedding max length exceeded.")
        x = x + self.pos_embed[:, :x.size(1), :]

        # Transformer 编码
        x = self.transformer(x)              # [B, N, C]
        x = x.mean(dim=1)                    # [B, C]

        # 融合 head pose
        if self.use_headpose and head_pose is not None:
            x = torch.cat([x, head_pose], dim=1)  # [B, C+2]

        return self.regressor(x)             # [B, 2]


# 包装类（推荐用于从 config 中创建模型）
def build_model_from_config(config):
    return GazeTransformer(
        feature_dim=config['MODEL']['TRANSFORMER']['FEATURE_DIM'],
        transformer_layers=config['MODEL']['TRANSFORMER']['LAYERS'],
        transformer_heads=config['MODEL']['TRANSFORMER']['HEADS'],
        use_headpose=config['MODEL']['TRANSFORMER']['USE_HEADPOSE']
    )
