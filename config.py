# config.py
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Gaze Estimation Project Config")

    # === 数据路径 ===
    parser.add_argument('--h5_path', type=str,
                        default='D:/gaze_estimation/data/MPIIGaze_processed/MPIIGaze.h5',
                        help='Path to the MPIIGaze.h5 file')

    # === 被试划分 ===
    parser.add_argument('--test_ids', type=int, nargs='+', default=[0],
                        help='Subject IDs to use as test set (e.g., --test_ids 0 1)')

    parser.add_argument('--eval_ids', type=int, nargs='+', default=[1],
                        help='Subject IDs to use as test set (e.g., --test_ids 0 1)')

    # === 图像预处理 ===
    parser.add_argument('--image_size', type=int, nargs=2, default=[96, 160],
                        help='Input image size as [height, width]')

    # === 模型结构 ===,erruh
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--transformer_layers', type=int, default=6)
    parser.add_argument('--transformer_heads', type=int, default=4)
    parser.add_argument('--use_headpose', action='store_true')
    parser.set_defaults(use_headpose=True)
    parser.add_argument('--mirror_video', action='store_true', help='Flip video horizontally (mirror effect)')

    # === 训练参数 ===
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=42)

    # === 学习率调度器相关参数 ===
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['none', 'step', 'cosine'],
                        help='学习率调度器类型（none / step / cosine）')
    parser.add_argument('--step_size', type=int, default=5,
                        help='如果使用StepLR，每隔多少个epoch衰减一次')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='StepLR中的衰减因子')

    # === 设备参数 ===
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (e.g., "cuda" or "cpu")')

    # === 模型加载（预训练） ===
    parser.add_argument('--load_pretrained', action='store_true',
                        help='Whether to load pretrained weights')
    parser.add_argument('--pretrained_path', type=str,
                        default='D:/gaze_estimation/checkpoints/model_epoch16_pre.pt',
                        help='Path to pretrained .pt file')

    # === 模型保存与日志 ===
    parser.add_argument('--log_dir', type=str, default='runs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory for saving model checkpoints')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save model every N epochs')

    # === 测试参数 ===
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Checkpoint file to load for evaluation or testing')

    return parser.parse_args()
