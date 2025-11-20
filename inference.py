import os
import cv2
import mmcv
import torch
import numpy as np
import matplotlib.pyplot as plt

from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

# 1. MMPose 모듈 및 사용자 정의 모듈 등록
# custom_imports에 정의된 모듈들을 인식하기 위해 필수입니다.
register_all_modules()

def run_inference(config_file, checkpoint_file, img_path, out_file='result.jpg'):
    # 2. 모델 초기화
    # config_file: satellite/rtmpose.py 경로
    # checkpoint_file: 학습된 .pth 파일 경로
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = init_model(config_file, checkpoint_file, device=device)

    # 3. 이미지 로드
    img = mmcv.imread(img_path)
    
    # 4. Bounding Box 설정
    # Top-down 방식이므로 bbox가 필요합니다. 
    # 인공위성 전체가 대상이므로 이미지 전체 크기를 bbox로 설정합니다.
    h, w, _ = img.shape
    bbox = [0, 0, w, h]  # [x, y, w, h] 형식이 아니라 [x1, y1, x2, y2] 형식이 필요할 수 있으나
                         # mmpose 구버전/신버전 호환성을 위해 xyxy로 변환하여 전달하는 것이 안전합니다.
    bbox_xyxy = np.array([0, 0, w, h])

    # 5. 인퍼런스 수행
    # bboxes는 (N, 4) 형태의 numpy array 또는 list를 받습니다.
    results = inference_topdown(model, img, bboxes=bbox_xyxy[None])
    
    # 결과 데이터 샘플 (첫 번째 bbox에 대한 결과)
    pred_instance = results[0].pred_instances

    # 6. 결과 출력 (Keypoint 좌표 및 Score)
    keypoints = pred_instance.keypoints[0]
    scores = pred_instance.keypoint_scores[0]

    print(f"\n=== Inference Results for {os.path.basename(img_path)} ===")
    for i, (kpt, score) in enumerate(zip(keypoints, scores)):
        print(f"Keypoint {i}: Coords=({kpt[0]:.2f}, {kpt[1]:.2f}), Score={score:.4f}")

    # 7. 시각화 (Skeleton 그리기)
    # 모델의 설정(cfg)에 있는 visualizer를 빌드하여 사용합니다.
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta)

    # 이미지에 결과 그리기
    visualizer.add_datasample(
        'result',
        img,
        data_sample=results[0],
        draw_gt=False,
        draw_heatmap=False, # SimCC 모델은 일반적인 2D Heatmap 시각화를 지원하지 않을 수 있음
        draw_bbox=True,
        show_kpt_idx=True,
        skeleton_style='mmpose',
        show=False,
        out_file=out_file,
        kpt_thr=0.3
    )
    print(f"Visualization saved to: {out_file}")

    return keypoints, scores

# --- 실행 설정 ---
if __name__ == '__main__':
    # 경로 설정 (사용자 환경에 맞게 수정 필요)
    CONFIG_FILE = 'satellite/rtmpose.py'  # 제공해주신 config 파일 경로
    CHECKPOINT_FILE = '/workspace/epoch_30.pth' # 학습된 체크포인트 경로 (예시)
    IMAGE_PATH = '/workspace/speedplusv2/train/000001.jpg' # 테스트할 이미지 경로
    
    # 실행
    try:
        run_inference(CONFIG_FILE, CHECKPOINT_FILE, IMAGE_PATH)
    except FileNotFoundError as e:
        print(f"오류 발생: {e}")
        print("체크포인트 파일 경로와 이미지 경로를 확인해주세요.")
    except Exception as e:
        print(f"실행 중 오류 발생: {e}")
        # custom_imports 문제일 경우 경로 추가가 필요할 수 있습니다.
        import sys
        sys.path.append(os.getcwd())