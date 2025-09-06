import cv2
import torch
import numpy as np
import face_alignment
from scipy.spatial.transform import Rotation as R

from config import get_args
from model.gaze_model import GazeTransformer
from utils.transforms import build_transform
import os





def get_eye_centers(landmarks):
    # 36–41 左眼， 42–47 右眼 (dlib index)
    left = landmarks[36:42]
    right = landmarks[42:48]
    return np.mean(left, axis=0).astype(int), np.mean(right, axis=0).astype(int)


def estimate_head_pose(landmarks, size):
    # SolvePnP with generic 3D landmarks (eye corners, nose, chin)
    image_pts = np.array([landmarks[30], landmarks[8], landmarks[36], landmarks[45], landmarks[48], landmarks[54]],
                         dtype='double')
    model_pts = np.array([[0.0, 0.0, 0.0],
                          [0.0, -330.0, -65.0],
                          [-225.0, 170.0, -135.0],
                          [225.0, 170.0, -135.0],
                          [-150.0, -150.0, -125.0],
                          [150.0, -150.0, -125.0]], dtype='double')
    focal = size[1]
    center = (size[1] / 2, size[0] / 2)
    cam = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]])
    dist = np.zeros((4, 1))
    _, rvec, tvec = cv2.solvePnP(model_pts, image_pts, cam, dist)
    rmat, _ = cv2.Rodrigues(rvec)
    rot = R.from_matrix(rmat)
    pitch, yaw, roll = rot.as_euler('xyz', degrees=True)
    return pitch, yaw


def draw_vector(img, origin, pitch, yaw, base_length=600):
    """
    Draws a perspective-corrected gaze vector from pitch and yaw.
    Arrow length is modulated to reflect 3D direction.
    """
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)

    # 3D gaze direction unit vector
    dx = np.cos(pitch_rad) * np.sin(yaw_rad)
    dy = -np.sin(pitch_rad)  # screen y-axis down is positive
    dz = -np.cos(pitch_rad) * np.cos(yaw_rad)  # z forward (into screen)


    perspective_factor = 1 + 0.5 * abs(dx) + 0.5 * abs(dy)
    length = base_length * perspective_factor

    end_x = int(origin[0] + length * dx)
    end_y = int(origin[1] + length * dy)

    cv2.arrowedLine(img, origin, (end_x, end_y), (0, 255, 0), 2, tipLength=0.25)


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = GazeTransformer(args.feature_dim, args.transformer_layers, args.transformer_heads, use_headpose=True).to(
        device)
    ckpt = "D:/gaze_estimation/checkpoints/model_epoch7_pre.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    transform = build_transform(image_size=tuple(args.image_size))

    cap = cv2.VideoCapture(0)

    output_path = "D:/gaze_estimation/output/gaze_demo.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'XVID' 保存为.avi
    fps = 20.0
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    ret, frame = cap.read()
    if not ret:
        print("❌ 摄像头读取失败")
        return

    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # N=468 landmarks for Mediapipe-style face alignment
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret: break

            if args.mirror_video:
                frame = cv2.flip(frame, 1)  # ← 镜像处理

            landmarks = fa.get_landmarks(frame)
            if landmarks:
                lm = landmarks[0]
                left_c, right_c = get_eye_centers(lm)
                pitch_h, yaw_h = estimate_head_pose(lm, frame.shape)

                for origin in [left_c, right_c]:
                    x, y = origin
                    eye_img = frame[y - 30:y + 30, x - 40:x + 40]
                    if eye_img.shape[0] < 10: continue
                    inp = transform(eye_img).unsqueeze(0).to(device)
                    head_t = torch.tensor([[pitch_h, yaw_h]], dtype=torch.float32).to(device)
                    p = model(inp, head_t).cpu().numpy()[0]
                    draw_vector(frame, origin, p[0], p[1])

                cv2.putText(frame, f"Head pitch:{pitch_h:.1f}, yaw:{yaw_h:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 1)

            cv2.imshow("Gaze Demo", frame)
            if cv2.waitKey(1) == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
