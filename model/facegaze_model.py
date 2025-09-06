import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class GazeTransformer(nn.Module):
    def __init__(self, feature_dim=256, transformer_layers=4, transformer_heads=4, use_headpose=True,
                 use_landmark=True):
        super().__init__()

        self.use_headpose = use_headpose
        self.use_landmark = False # use_landmark
        self.feature_dim = feature_dim

        # CNN Backbone（保留原结构，仅到 layer3）
        base_cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cnn_backbone = nn.Sequential(
            base_cnn.conv1,
            base_cnn.bn1,
            base_cnn.relu,
            base_cnn.maxpool,
            base_cnn.layer1,
            base_cnn.layer2,
            base_cnn.layer3  # 输出特征维度为 256，stride=16
        )

        # 1×1 卷积进行通道投影
        self.conv_proj = nn.Conv2d(256, feature_dim, kernel_size=1)

        # 位置编码（784 + 1 = 785 个 token，预留 1 个 landmark token）
        self.max_token_num = 784
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_token_num, feature_dim))

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=transformer_heads, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # 回归头（只接收 transformer mean pooled + head_pose）
        headpose_dim = 2 if use_headpose else 0
        regressor_input_dim = feature_dim + headpose_dim
        self.regressor = nn.Sequential(
            nn.Linear(regressor_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 2)
        )
        print(f"[Init] use_headpose={self.use_headpose}, use_landmark={self.use_landmark}")

    def forward(self, x, head_pose=None, landmark=None):
        x = self.cnn_backbone(x)
        x = self.conv_proj(x)  # [B, C, H, W]
        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)  # [B, N, C], N = H*W = 784

        # 位置编码
        x = x + self.pos_embed[:, :x.size(1), :]

        # Transformer 编码器
        x = self.transformer(x)  # [B, N+1, C] or [B, N, C]

        # Mean pool所有 patch token（可选择 skip 第0个 landmark token）
        if self.use_landmark:
            x = x[:, 1:, :].mean(dim=1)  # [B, C] 忽略 landmark token
        else:
            x = x.mean(dim=1)  # [B, C]

        # 拼接 head pose
        if self.use_headpose and head_pose is not None:
            x = torch.cat([x, head_pose], dim=1)

        return self.regressor(x)


# 用 config 创建模型
def build_model_from_config(config):
    return GazeTransformer(
        feature_dim=config['MODEL']['TRANSFORMER']['FEATURE_DIM'],
        transformer_layers=config['MODEL']['TRANSFORMER']['LAYERS'],
        transformer_heads=config['MODEL']['TRANSFORMER']['HEADS'],
        use_headpose=config['MODEL']['TRANSFORMER']['USE_HEADPOSE'],
        use_landmark=config['MODEL']['TRANSFORMER'].get('USE_LANDMARK', True)  # ✅ 加这一行
    )