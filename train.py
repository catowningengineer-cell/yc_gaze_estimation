# train.py

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import get_args
from model.gaze_model import GazeTransformer
from dataset.mpiigaze_dataset import MPIIGazeDataset
from utils.transforms import build_transform
from utils.trainer import train_one_epoch, evaluate, load_pretrained_model


def main():
    args = get_args()
    args.use_headpose = True

    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # === 构建模型 ===
    model = GazeTransformer(
        feature_dim=args.feature_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        use_headpose=args.use_headpose
    ).to(device)

    # === 加载预训练权重 ===
    if args.load_pretrained and args.pretrained_path is not None:
        model = load_pretrained_model(model, args.pretrained_path)

    # === 数据划分（按被试编号） ===
    all_ids = list(range(15))
    test_ids = args.test_ids
    eval_ids = args.eval_ids
    train_ids = [i for i in all_ids if i not in test_ids]

    transform = build_transform(image_size=tuple(args.image_size))

    train_dataset = MPIIGazeDataset(args.h5_path, subject_ids=train_ids, transform=transform)
    val_dataset = MPIIGazeDataset(args.h5_path, subject_ids=test_ids, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # === 优化器 ===
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # === 学习率调度器（可选） ===
    if args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1.5 * args.epochs)
    else:
        scheduler = None

    # === 日志和保存目录 ===
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mse = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_angle = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch}/{args.epochs}] Train Loss: {train_loss:.4f}|Train MSE: {train_mse:.4f}| Val Loss: {val_loss:.4f} | Angle: {val_angle:.2f}°")
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)

        if epoch % args.save_freq == 0 or val_loss < best_val_loss or val_angle < 14.5:
            save_path = os.path.join(args.save_dir, f"model_epoch{epoch}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            best_val_loss = min(best_val_loss, val_loss)

    writer.close()


if __name__ == "__main__":
    main()
