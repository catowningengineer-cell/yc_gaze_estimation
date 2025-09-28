# facegaze_config.py
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="MPIIFaceGaze Fine-tuning Config")

    # === 数据路径 ===
    parser.add_argument('--h5_path', type=str,
                        default='/root/code/gaze_estimation/data/mpiifacegaze_processed/MPIIFaceGaze.h5',
                        help='Path to the MPIIFaceGaze.h5 file')

    # === 被试划分 ===
    parser.add_argument('--participants', type=int, nargs='+', default=list(range(10)),
                        help='Participant IDs to use (e.g., 0–9 for training)')

    # === 图像输入尺寸 ===
    parser.add_argument('--image_size', type=int, nargs=2, default=[112, 112],
                        help='Input image size [height, width]')

    # === 模型结构参数 ===
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--transformer_layers', type=int, default=6)
    parser.add_argument('--transformer_heads', type=int, default=4)
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate in Transformer encoder')
    parser.add_argument('--use_headpose', action='store_true')

    # === 训练参数 ===
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=41)

    # === 学习率调度器相关参数 ===
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['none', 'step', 'cosine'],
                        help='学习率调度器类型（none / step / cosine）')
    parser.add_argument('--step_size', type=int, default=10,
                        help='如果使用StepLR，每隔多少个epoch衰减一次')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='StepLR中的衰减因子')
    # === Warmup 参数 ===
    parser.add_argument('--use_warmup', action='store_true', help='是否使用 warmup 策略')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup 持续的 epoch 数')

    # === 设备参数 ===
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (e.g., "cuda" or "cpu")')

    # === 模型加载（预训练） ===
    parser.add_argument('--load_pretrained', action='store_true',
                        help='Whether to load pretrained weights')
    parser.add_argument('--pretrained_path', type=str,
                        default='/root/code/gaze_estimation/checkpoints/model_epoch16_pre.pt',
                        help='Path to pretrained .pt file')

    # === 日志与模型保存 ===
    parser.add_argument('--log_dir', type=str,
                        default='/root/autodl-tmp/yutong_robotics/face_logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--save_dir', type=str,
                        default='/root/autodl-tmp/yutong_robotics/face_checkpoints',
                        help='Directory for saving model checkpoints')
    parser.add_argument('--save_freq', type=int, default=5)

    # === 测试用 ===
    parser.add_argument('--ckpt_path', type=str, default=None)

    return parser.parse_args()

