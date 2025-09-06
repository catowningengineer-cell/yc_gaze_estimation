import argparse
import pathlib
import h5py
import numpy as np
import tqdm


def save_one_person(person_id: str, data_dir: pathlib.Path, output_path: pathlib.Path) -> None:
    mat_path = data_dir / f'{person_id}.mat'
    if not mat_path.exists():
        print(f"[Warning] File not found: {mat_path}")
        return

    with h5py.File(mat_path, 'r') as f:
        # 读取数据并进行格式转换： (N, C, H, W) -> (N, H, W, C)
        images = f['Data/data'][()]  # uint8, shape: (3000, 3, 36, 60)
        images = images.transpose(0, 2, 3, 1).astype(np.uint8)  # -> (3000, 36, 60, 3)

        # 读取 gaze 和 pose 标签： pitch, yaw 结构
        labels = f['Data/label'][()]  # shape: (3000, 4)
        gazes = labels[:, :2].astype(np.float32)  # (pitch, yaw)
        poses = labels[:, 2:4].astype(np.float32)  # (pitch, yaw)
        landmarks = labels[:, 4:16].astype(np.float32)

    # 写入到 HDF5 文件
    with h5py.File(output_path, 'a') as f_out:
        f_out.create_dataset(f'{person_id}/image', data=images)
        f_out.create_dataset(f'{person_id}/pose', data=poses)
        f_out.create_dataset(f'{person_id}/gaze', data=gazes)
        f_out.create_dataset(f'{person_id}/landmark', data=landmarks)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='D:/gaze_estimation/data/MPIIFaceGaze_normalized',
                        help='Path to original .mat MPIIFaceGaze directory')
    parser.add_argument('--output-dir', '-o', type=str, default='D:/gaze_estimation/data/MPIIFaceGaze_processed',
                        help='Path to save output .h5 file')
    args = parser.parse_args()

    data_dir = pathlib.Path(args.dataset).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'MPIIFaceGaze.h5'
    if output_path.exists():
        raise FileExistsError(f"{output_path} already exists. Remove it manually if you want to regenerate.")

    for pid in tqdm.tqdm(range(15), desc='Processing MPIIFaceGaze participants'):
        person_id = f'p{pid:02}'
        save_one_person(person_id, data_dir, output_path)


if __name__ == '__main__':
    main()
