#!/usr/bin/env python

import argparse
import pathlib
import cv2
import h5py
import numpy as np
import pandas as pd
import scipy.io
import tqdm


def convert_pose(vector: np.ndarray) -> np.ndarray:
    rot = cv2.Rodrigues(np.array(vector).astype(np.float32))[0]
    vec = rot[:, 2]
    pitch = np.arcsin(vec[1])
    yaw = np.arctan2(vec[0], vec[2])
    return np.array([pitch, yaw], dtype=np.float32)


def convert_gaze(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    return np.array([pitch, yaw], dtype=np.float32)


def get_eval_info(person_id: str, eval_dir: pathlib.Path) -> pd.DataFrame:
    eval_path = eval_dir / f'{person_id}.txt'
    df = pd.read_csv(eval_path, delimiter=' ', header=None, names=['path', 'side'])
    df['day'] = df.path.apply(lambda p: p.split('/')[0])
    df['filename'] = df.path.apply(lambda p: p.split('/')[1])
    return df.drop(columns=['path'])


def save_one_person(person_id: str, data_dir: pathlib.Path, eval_dir: pathlib.Path, output_path: pathlib.Path) -> None:
    left_images, left_poses, left_gazes = {}, {}, {}
    right_images, right_poses, right_gazes = {}, {}, {}
    filenames = {}

    person_dir = data_dir / person_id
    for path in sorted(person_dir.glob('*')):
        mat_data = scipy.io.loadmat(str(path), struct_as_record=False, squeeze_me=True)
        data = mat_data['data']
        day = path.stem

        left_images[day] = np.array(data.left.image)
        left_poses[day] = np.array(data.left.pose)
        left_gazes[day] = np.array(data.left.gaze)

        right_images[day] = np.array(data.right.image)
        right_poses[day] = np.array(data.right.pose)
        right_gazes[day] = np.array(data.right.gaze)

        filenames[day] = mat_data['filenames']
        if not isinstance(filenames[day], np.ndarray):
            for k in [left_images, left_poses, left_gazes, right_images, right_poses, right_gazes]:
                k[day] = np.array([k[day]])
            filenames[day] = np.array([filenames[day]])

    df = get_eval_info(person_id, eval_dir)

    images, poses, gazes = [], [], []
    for _, row in df.iterrows():
        day, fname, side = row.day, row.filename, row.side
        idx = np.where(filenames[day] == fname)[0][0]
        if side == 'left':
            image = left_images[day][idx]
            pose = convert_pose(left_poses[day][idx])
            gaze = convert_gaze(left_gazes[day][idx])
        else:
            image = right_images[day][idx][:, ::-1]  # 水平翻转
            pose = convert_pose(right_poses[day][idx]) * np.array([1, -1])
            gaze = convert_gaze(right_gazes[day][idx]) * np.array([1, -1])
        images.append(image)
        poses.append(pose)
        gazes.append(gaze)

    images = np.array(images, dtype=np.uint8)
    poses = np.array(poses, dtype=np.float32)
    gazes = np.array(gazes, dtype=np.float32)

    with h5py.File(output_path, 'a') as f_out:
        f_out.create_dataset(f'{person_id}/image', data=images)
        f_out.create_dataset(f'{person_id}/pose', data=poses)
        f_out.create_dataset(f'{person_id}/gaze', data=gazes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the MPIIGaze root directory (contains Data/, Evaluation Subset/)')
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                        help='Where to save processed MPIIGaze.h5 file')
    args = parser.parse_args()

    dataset_dir = pathlib.Path(args.dataset)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'MPIIGaze.h5'
    if output_path.exists():
        raise FileExistsError(f'{output_path} already exists. ')

    data_dir = dataset_dir / 'Data' / 'Normalized'
    eval_dir = dataset_dir / 'Evaluation Subset' / 'sample list for eye image'

    for pid in tqdm.tqdm(range(15), desc='Processing participants'):
        person_id = f'p{pid:02}'
        save_one_person(person_id, data_dir, eval_dir, output_path)


if __name__ == '__main__':
    main()
