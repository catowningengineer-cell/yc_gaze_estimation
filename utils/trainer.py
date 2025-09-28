import torch
import torch.nn.functional as F

def _pitchyaw_to_vec(p: torch.Tensor) -> torch.Tensor:
    pitch, yaw = p[:, 0], p[:, 1]
    x = torch.cos(pitch) * torch.sin(yaw)
    y = torch.sin(pitch)
    z = torch.cos(pitch) * torch.cos(yaw)
    v = torch.stack([x, y, z], dim=1)
    return F.normalize(v, dim=1)

def _as_unit_vec(t: torch.Tensor) -> torch.Tensor:
    if t.dim() != 2 or t.size(1) not in (2, 3):
        raise ValueError(f"Unexpected tensor shape {tuple(t.shape)}; expect [B,2] or [B,3].")
    if t.size(1) == 2:
        return _pitchyaw_to_vec(t)
    return F.normalize(t, dim=1)

@torch.no_grad()
def angular_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    v1, v2 = _as_unit_vec(pred), _as_unit_vec(gt)
    dot = (v1 * v2).sum(-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(dot) * (180.0 / math.pi)

def angular_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    v1, v2 = _as_unit_vec(pred), _as_unit_vec(gt)
    dot = (v1 * v2).sum(-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(dot).mean()


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_mse = 0
    alpha = 0.001

    for batch in dataloader:
        eye = batch['image'].to(device)
        head_pose = batch['head_pose'].to(device)
        gaze = batch['gaze'].to(device)

        pred = model(eye, head_pose)
        mse = F.mse_loss(pred, gaze)
        angle = angular_loss(pred, gaze)
        loss = mse + alpha * angle

        optimizer.zero_grad()
        loss.backward()
        mse.backward
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()

    avg_mse = total_mse / len(dataloader)
    avg_loss = total_loss / len(dataloader)
    return avg_loss, avg_mse


def evaluate(model, dataloader, device):
    alpha = 0.001
    model.eval()
    total_mse = 0
    total_angle_loss = 0
    all_angle_errors = []

    with torch.no_grad():
        for batch in dataloader:
            eye = batch['image'].to(device)
            gaze = batch['gaze'].to(device)
            head_pose = batch['head_pose'].to(device)

            pred = model(eye, head_pose)
            mse = F.mse_loss(pred, gaze)
            angle_loss = angular_loss(pred, gaze)

            angles = angular_error(pred, gaze)
            all_angle_errors.extend(angles.cpu().numpy())

            total_mse += mse.item()
            total_angle_loss += angle_loss.item()

    avg_mse = total_mse / len(dataloader)
    avg_angle_loss = total_angle_loss / len(dataloader)
    avg_angle = sum(all_angle_errors) / len(all_angle_errors)

    print(f"[Eval] MSE Loss: {avg_mse:.4f} | Angular Loss: {avg_angle_loss:.4f} | Angular Error: {avg_angle:.2f}°")
    return avg_mse, avg_angle


def load_pretrained_model(model, pretrained_path):
    import torch

    print(f"Loading pretrained weights from: {pretrained_path}")
    try:
        state_dict = torch.load(pretrained_path, map_location='cpu')

        # Check whether this is a full checkpoint or just a state_dict
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            print("Detected checkpoint format with 'model_state_dict'")
            state_dict = state_dict['model_state_dict']
        else:
            print("Detected pure state_dict format")

        # Filter out incompatible parameters (e.g., pos_embed and regressor)
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if 'pos_embed' in k:
                print(f"Skipping incompatible key (pos_embed): {k}")
                continue
            if 'regressor' in k:
                print(f"Skipping incompatible key (regressor): {k}")
                continue
            if k in model.state_dict():
                filtered_state_dict[k] = v

        # Load the compatible weights
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        print("Model loaded successfully (except for skipped layers).")
        if missing_keys:
            print(f"Missing keys not loaded from checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys in checkpoint: {unexpected_keys}")

    except Exception as e:
        print(f"Failed to load pretrained model: {e}")

    return model

