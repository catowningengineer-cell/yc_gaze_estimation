from torchvision import transforms


def build_facegaze_transform(image_size=(112, 112)):
    """
    为 MPIIFaceGaze 构建图像预处理流程：
    - Resize 到给定分辨率（默认 448x448）
    - 转换为 tensor
    - 标准化到 [-1, 1] 区间
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.40],
                             std=[0.229, 0.224, 0.225])
    ])
