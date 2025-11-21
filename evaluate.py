import os
import json
import cv2
import numpy as np
import torch
import math
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules

# --- 설정 ---
# 경로를 실제 환경에 맞게 수정해주세요.
CONFIG_FILE = 'satellite/rtmpose.py'
CHECKPOINT_FILE = '/workspace/epoch_30.pth' # 학습된 가중치 경로
ANNO_FILE = '/workspace/speedplusv2/synthetic/validation.json' # GT json 경로
IMG_ROOT = '/workspace/speedplusv2/val/' # 이미지 폴더 (validation.json의 filename이 상대경로일 경우 root 필요)
MODEL_3D_POINTS_FILE = '/workspace/speedplusv2/tangoPoints.mat' # 3D 포인트 파일 경로
CAMERA_FILE = '/workspace/speedplusv2/camera.json' # 카메라 파라미터 파일 (없으면 아래 K 행렬 직접 수정)

# ------------------------------------------------------------------------------
# Helper Functions (spnv2/core/utils/postprocess.py 및 utils.py 참고)
# ------------------------------------------------------------------------------

def load_camera_intrinsics(camera_json_path):
    """카메라 내부 파라미터 로드 (파일이 없으면 SPEED+ 기본값 사용)"""
    if os.path.exists(camera_json_path):
        with open(camera_json_path) as f:
            cam = json.load(f)
        # 딕셔너리를 numpy array로 변환 등의 처리가 필요할 수 있음
        # 여기서는 편의상 SPEED+ Next Generation benchmark의 일반적인 값을 예시로 사용
        # 실제 camera.json 구조에 맞춰 파싱해야 함.
        K = np.array(cam['cameraMatrix'], dtype=np.float32)
        dist = np.array(cam['distCoeffs'], dtype=np.float32).flatten()
    else:
        print(f"Warning: {camera_json_path} not found. Using default SPEED+ intrinsics.")
        # SPEED+ 기본값 (예시, 실제 데이터셋에 맞게 수정 필수)
        # Nu=1920, Nv=1200
        fx = 0.0176  # m 단위 focal length? SPEED 데이터셋은 보통 픽셀 단위 K를 제공함.
        # 아래는 예시 값입니다. 실제 json 내용을 확인하여 K 행렬을 구성하세요.
        K = np.array([[17600, 0, 960],
                      [0, 17600, 600],
                      [0, 0, 1]], dtype=np.float32) 
        dist = np.zeros(5, dtype=np.float32)
        
    return K, dist

def load_tango_3d_keypoints(mat_path):
    """3D 모델 포인트 로드 (3x11 -> 11x3 변환)"""
    try:
        vertices = loadmat(mat_path)['tango3Dpoints'] # [3 x 11]
        corners3D = np.transpose(np.array(vertices, dtype=np.float32)) # [11 x 3]
        return corners3D
    except Exception as e:
        print(f"Error loading 3D points: {e}")
        # 테스트용 더미 데이터 (실제 실행 시 에러 발생하므로 파일 확인 필요)
        return np.zeros((11, 3), dtype=np.float32)

def solve_pnp_epnp(points_3D, points_2D, cameraMatrix, distCoeffs):
    """
    SOLVEPNP_EPNP를 사용하여 Pose 추정
    Returns:
        q (np.array): [w, x, y, z] quaternion (Scalar-first)
        t (np.array): [x, y, z] translation vector
    """
    if distCoeffs is None:
        distCoeffs = np.zeros((5, 1), dtype=np.float32)

    # 형상 맞춤 (N, 3), (N, 2)
    points_3D = np.ascontiguousarray(points_3D).reshape((-1, 1, 3))
    points_2D = np.ascontiguousarray(points_2D).reshape((-1, 1, 2))

    # EPnP 실행
    success, rvec, tvec = cv2.solvePnP(
        points_3D,
        points_2D,
        cameraMatrix,
        distCoeffs,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not success:
        return None, None

    # Rotation Vector -> Rotation Matrix -> Quaternion
    R_mat, _ = cv2.Rodrigues(rvec)
    q = R.from_matrix(R_mat).as_quat() # scipy는 [x, y, z, w] (scalar-last)

    # [x, y, z, w] -> [w, x, y, z] (Scalar-first)로 변환 (spnv2 관례 따름)
    q_scalar_first = q[[3, 0, 1, 2]]
    
    return q_scalar_first, np.squeeze(tvec)

def compute_errors(q_pred, t_pred, q_gt, t_gt):
    """Translation 및 Rotation 오차 계산"""
    # 1. Translation Error (m) -> Euclidean distance
    t_err = np.linalg.norm(t_pred - t_gt)

    # 2. Rotation Error (deg) -> 2 * arccos(|<q1, q2>|)
    # q는 모두 normalized 상태여야 함
    q_pred = q_pred / np.linalg.norm(q_pred)
    q_gt = q_gt / np.linalg.norm(q_gt)
    
    # 내적 (Dot product)
    dot = np.abs(np.dot(q_pred, q_gt))
    # 수치적 오차로 1.0을 넘는 경우 클리핑
    dot = np.clip(dot, -1.0, 1.0)
    
    rad_err = 2 * np.arccos(dot)
    deg_err = np.rad2deg(rad_err)

    return t_err, deg_err

# ------------------------------------------------------------------------------
# Main Evaluation Loop
# ------------------------------------------------------------------------------

def main():
    # 1. 모듈 등록 및 모델 로드
    register_all_modules()
    
    print(f"Loading model from {CONFIG_FILE}...")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=device)

    # 2. 데이터 로드 (3D Points, Camera, Annotations)
    points_3d = load_tango_3d_keypoints(MODEL_3D_POINTS_FILE)
    K, dist = load_camera_intrinsics(CAMERA_FILE)
    
    print(f"Loading annotations from {ANNO_FILE}...")
    with open(ANNO_FILE, 'r') as f:
        annotations = json.load(f)
    
    # annotations가 list 형태라고 가정 (COCO 포맷이 아닐 경우 구조 확인 필요)
    # 만약 COCO 포맷이라면 annotations['images'], annotations['annotations'] 파싱 필요
    # 여기서는 사용자가 언급한 구조(filename, r_..., q_...)를 가진 리스트라고 가정
    if isinstance(annotations, dict) and 'images' in annotations:
         # COCO format 처리 (필요시 구현)
         print("Note: Loaded standard COCO format JSON. Extracting relevant list...")
         # 실제로는 validation.json 구조에 따라 루프를 다르게 짜야 합니다.
         # 사용자가 명시한 키가 바로 있는 리스트 구조로 가정합니다.
         data_list = annotations['images'] if 'images' in annotations else [] # 수정 필요 가능성 있음
         # SPEED+ 데이터셋 구조에 맞게 매핑
         # 보통 SPEED+ json은 list of dicts 형태임.
         pass 
    else:
        data_list = annotations

    t_errors = []
    q_errors = []
    
    print(f"Starting evaluation on {len(data_list)} images...")
    
    for item in tqdm(data_list):
        # 파일 경로 구성
        img_name = item['filename']

        if img_name.startswith('img'):
            img_name = img_name[3:]

        # json에 전체 경로가 있을 수도 있고 파일명만 있을 수도 있음
        if os.path.isabs(img_name):
            img_path = img_name
        else:
            img_path = os.path.join(IMG_ROOT, img_name)
        
        # GT 로드
        t_gt = np.array(item['r_Vo2To_vbs_true'], dtype=np.float32)
        q_gt = np.array(item['q_vbs2tango_true'], dtype=np.float32) # [w, x, y, z] 가정

        # 이미지 로드 확인
        if not os.path.exists(img_path):
            continue

        # 3. Inference (2D Keypoints)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        bbox = np.array([0, 0, w, h]) # 전체 이미지 bbox

        results = inference_topdown(model, img, bboxes=bbox[None])
        pred_instances = results[0].pred_instances
        
        # (N, 2) keypoints
        kpts_2d = pred_instances.keypoints[0] 
        scores = pred_instances.keypoint_scores[0]

        # 4. Filter Keypoints (Optional: spnv2/postprocess.py 로직)
        # 신뢰도가 낮은 포인트는 제외하고 PnP를 풀 수도 있음.
        # 여기서는 11개 포인트 모두 사용한다고 가정 (EPnP는 점이 많을수록 좋음)
        # 필요하다면:
        # valid_mask = scores > 0.3
        # if np.sum(valid_mask) < 4: continue
        # kpts_2d = kpts_2d[valid_mask]
        # points_3d_batch = points_3d[valid_mask]
        
        points_3d_batch = points_3d # 전체 11개 사용

        # 5. Solve PnP (EPnP)
        q_pred, t_pred = solve_pnp_epnp(points_3d_batch, kpts_2d, K, dist)

        if q_pred is not None:
            # 6. Compare with GT
            t_err, q_err = compute_errors(q_pred, t_pred, q_gt, t_gt)
            
            t_errors.append(t_err)
            q_errors.append(q_err)
            
            # (옵션) 개별 결과 출력
            # print(f"{img_name} | T_err: {t_err:.4f} m | Q_err: {q_err:.4f} deg")

    # 7. 최종 결과 출력
    mean_t_err = np.mean(t_errors)
    mean_q_err = np.mean(q_errors)
    
    print("\n=== Evaluation Results ===")
    print(f"Total Images Evaluated: {len(t_errors)}")
    print(f"Mean Translation Error: {mean_t_err:.6f} m")
    print(f"Mean Rotation Error:    {mean_q_err:.6f} deg")
    
    # SPEED Score (Example metric)
    # Normalized error sum (Need valid normalization factors)
    print("==========================")

if __name__ == '__main__':
    main()