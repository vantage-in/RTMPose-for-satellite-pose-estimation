import os
import random
import cv2
import mmcv
import torch
import numpy as np
from pathlib import Path

from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules
from mmpose.registry import VISUALIZERS

# 1. MMPose 모듈 및 사용자 정의 모듈 등록
register_all_modules()

def run_inference_on_image(model, img_path, out_dir, visualizer):
    """
    단일 이미지에 대해 인퍼런스를 수행하고 결과를 저장합니다.
    """
    # 이미지 로드
    img = mmcv.imread(img_path)
    
    # Bounding Box 설정 (이미지 전체)
    h, w, _ = img.shape
    bbox_xyxy = np.array([0, 0, w, h])

    # 인퍼런스 수행
    results = inference_topdown(model, img, bboxes=bbox_xyxy[None])
    
    # 결과 데이터 (첫 번째 bbox)
    pred_instance = results[0].pred_instances
    keypoints = pred_instance.keypoints[0]
    scores = pred_instance.keypoint_scores[0]

    # 결과 파일명 생성
    img_name = os.path.basename(img_path)
    out_path = os.path.join(out_dir, f"vis_{img_name}")

    # 시각화 및 저장
    visualizer.add_datasample(
        'result',
        img,
        data_sample=results[0],
        draw_gt=False,
        draw_heatmap=False,
        draw_bbox=True,
        show_kpt_idx=True,
        skeleton_style='mmpose',
        show=False,
        out_file=out_path,
        kpt_thr=0.3
    )
    
    return keypoints, scores, out_path

def process_folder(config_file, checkpoint_file, input_folder, output_folder, sample_num=5):
    """
    폴더 내 이미지를 샘플링하여 일괄 처리합니다.
    """
    # 2. 모델 초기화 (한 번만 수행)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing model on {device}...")
    model = init_model(config_file, checkpoint_file, device=device)
    
    # Visualizer 초기화
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta)

    # 3. 이미지 리스트 확보 및 샘플링
    # 지원하는 이미지 확장자
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    all_images = [
        os.path.join(input_folder, f) 
        for f in os.listdir(input_folder) 
        if f.lower().endswith(valid_extensions)
    ]

    if not all_images:
        print(f"No images found in {input_folder}")
        return

    # 샘플링 개수 조정 (전체 이미지보다 요청 개수가 많으면 전체 사용)
    num_to_sample = min(len(all_images), sample_num)
    sampled_images = random.sample(all_images, num_to_sample)
    
    print(f"\nProcessing {num_to_sample} images sampled from {len(all_images)} total images.")
    
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    # 4. 반복 처리
    for i, img_path in enumerate(sampled_images):
        print(f"[{i+1}/{num_to_sample}] Processing: {os.path.basename(img_path)}")
        try:
            kpts, scores, out_file = run_inference_on_image(model, img_path, output_folder, visualizer)
            print(f"   -> Saved to: {out_file}")
        except Exception as e:
            print(f"   -> Error processing {img_path}: {e}")

# --- 실행 설정 ---
if __name__ == '__main__':
    # 경로 설정 (사용자 환경에 맞게 수정 필요)
    CONFIG_FILE = 'satellite/rtmpose.py'               # 설정 파일 경로
    CHECKPOINT_FILE = '/workspace/epoch_30.pth'  # 체크포인트 경로
    
    INPUT_FOLDER = '/workspace/speedplusv2/val/'                 # 테스트 이미지가 있는 폴더
    OUTPUT_FOLDER = 'vis_results/'                     # 결과를 저장할 폴더
    SAMPLE_NUM = 30                                     # 샘플링할 이미지 개수 (전체를 원하면 매우 큰 수 입력)

    try:
        process_folder(CONFIG_FILE, CHECKPOINT_FILE, INPUT_FOLDER, OUTPUT_FOLDER, SAMPLE_NUM)
    except FileNotFoundError as e:
        print(f"오류: {e}")
        print("파일 경로를 다시 확인해주세요.")
    except Exception as e:
        print(f"실행 중 오류 발생: {e}")
        # 모듈 import 에러 시 현재 경로 추가
        import sys
        sys.path.append(os.getcwd())