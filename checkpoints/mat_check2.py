import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def visualize_sample(h5_path, person_id='p00', index=0, save_path=None):
    with h5py.File(h5_path, 'r') as f:
        image = f[f'{person_id}/image'][index]  # shape: (H, W, 3)
        landmark = f[f'{person_id}/landmark'][index]  # shape: (12,)

    # 转为 RGB 图像
    image = image.astype(np.uint8)

    # 拆分 landmark
    x_coords = landmark[0::2]
    y_coords = landmark[1::2]

    # 可视化
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.scatter(x_coords, y_coords, c='red', s=40, label='landmarks')
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        plt.text(x + 2, y + 2, f'{i + 1}', color='white', fontsize=8)
    plt.title(f'{person_id} - Index {index}')
    plt.axis('off')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    visualize_sample(
        h5_path='D:/gaze_estimation/data/MPIIFaceGaze_processed/MPIIFaceGaze.h5',
        person_id='p14',
        index=33,
        save_path=None  # 可设为 'output.png'
    )
