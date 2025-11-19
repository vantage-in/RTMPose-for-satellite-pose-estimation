import argparse
import os
import os.path as osp
import torch
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData
from mmpose.registry import DATASETS, MODELS
from mmpose.structures import PoseDataSample

# Headless 환경 지원
import matplotlib
matplotlib.use('Agg')

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Tensor Augmentation (Batch Augments)')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--output-dir', default='aug_check_results', help='output directory')
    parser.add_argument('--num-samples', type=int, default=10, help='number of samples to check')
    parser.add_argument('--device', default='cuda', help='device to run augmentation')
    args = parser.parse_args()
    return args

def denormalize(tensor, mean, std):
    """
    Normalize된 Tensor를 이미지(0-255, uint8, HWC, RGB)로 변환
    tensor: (C, H, W)
    """
    tensor = tensor.detach().cpu().clone()
    # (C, H, W) -> (H, W, C)
    tensor = tensor.permute(1, 2, 0)
    
    # 역정규화: input * std + mean
    mean = torch.tensor(mean).view(1, 1, 3)
    std = torch.tensor(std).view(1, 1, 3)
    
    img = tensor * std + mean
    img = img.clamp(0, 255).numpy().astype(np.uint8)
    return img

def main():
    args = parse_args()
    
    # 1. Config 로드
    print(f"Loading config: {args.config}")
    cfg = Config.fromfile(args.config)
    
    # mmpose 스코프 초기화 (레지스트리 접근용)
    init_default_scope(cfg.get('default_scope', 'mmpose'))

    # 2. 데이터셋 빌드
    print("Building dataset...")
    # train_dataloader 설정을 사용하여 데이터셋 빌드
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    
    # 3. Augmentor(CombinedAugmentation) 빌드
    print("Building Augmentor...")
    # Config에서 batch_augments 부분 추출
    if hasattr(cfg.model, 'data_preprocessor') and \
       'batch_augments' in cfg.model.data_preprocessor:
        
        aug_cfg = cfg.model.data_preprocessor.batch_augments[0]
        
        # CombinedAugmentation은 사용자 정의 모듈이므로, 
        # registry에 등록되어 있어야 함 (imports 실행 필요)
        # cfg 파일 상단에 import가 있다면 자동으로 되지만, 안전을 위해 확인
        try:
            augmentor = MODELS.build(aug_cfg)
        except Exception as e:
            print(f"Error building augmentor: {e}")
            print("커스텀 모듈 임포트 경로를 확인해주세요. (예: satellite.tensor_augmentor)")
            return
            
        augmentor.to(args.device)
        augmentor.train() # Train 모드여야 증강이 작동함
    else:
        print("Config에 batch_augments가 정의되지 않았습니다.")
        return

    # 전처리용 Mean/Std 가져오기 (DataPreprocessor 설정)
    mean_vals = cfg.model.data_preprocessor.mean
    std_vals = cfg.model.data_preprocessor.std
    
    # Tensor 계산을 위한 준비
    mean_tensor = torch.tensor(mean_vals).view(1, 3, 1, 1).to(args.device)
    std_tensor = torch.tensor(std_vals).view(1, 3, 1, 1).to(args.device)

    # 저장 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Start visualizing {args.num_samples} samples...")
    
    for i in range(min(len(dataset), args.num_samples)):
        # 데이터 하나 가져오기 (pipeline 통과 후)
        item = dataset[i]
        
        # item['inputs']는 보통 (3, H, W) 형태의 텐서 (값 범위는 파이프라인에 따라 다름)
        # MMPose PackPoseInputs는 보통 이미지를 Tensor로 바꾸지만 정규화는 안 할 수 있음.
        # rtmpose config를 보면 LoadImage -> Aug -> PackPoseInputs 순서임.
        # PackPoseInputs는 ToTensor 동작을 함 (0-255 or 0-1 check needed)
        # 보통 mmpose는 DataPreprocessor에서 정규화를 하므로, 여기 입력은 0~255 범위일 확률 높음.
        
        input_tensor = item['inputs'].float().to(args.device) # (3, H, W)
        
        # 배치 차원 추가 (1, 3, H, W)
        input_batch = input_tensor.unsqueeze(0)
        
        # --- DataPreprocessor 시뮬레이션 (정규화) ---
        # CombinedAugmentation은 Normalized 입력을 기대함 (코드 분석 결과)
        normalized_batch = (input_batch - mean_tensor) / std_tensor
        
        # --- CombinedAugmentation 적용 ---
        # data_samples는 None으로 넘겨도 Augmentor 로직상 문제 없으면 None (코드상 문제 없음)
        augmented_batch, _ = augmentor(normalized_batch, None)
        
        # --- 시각화 준비 ---
        # 1. 원본 (Augmentation 전, 정규화만 된 상태 -> 복원)
        img_org = denormalize(normalized_batch[0], mean_vals, std_vals)
        
        # 2. 결과 (Augmentation 후 -> 복원)
        img_aug = denormalize(augmented_batch[0], mean_vals, std_vals)
        
        # --- Plotting ---
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(img_org)
        axes[0].set_title("Original (Pre-processed)")
        axes[0].axis('off')
        
        axes[1].imshow(img_aug)
        axes[1].set_title("Augmented (Tensor Aug)")
        axes[1].axis('off')
        
        save_path = osp.join(args.output_dir, f"sample_{i:03d}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Saved: {save_path}")

    print("Done.")

if __name__ == '__main__':
    main()