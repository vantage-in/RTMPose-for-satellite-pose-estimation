# check_tensor_aug.py

import argparse
import os
import os.path as osp
import torch
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.registry import DATASETS, MODELS

# Headless 환경 지원
import matplotlib
matplotlib.use('Agg')

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Tensor Augmentation')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--output-dir', default='tensor_aug_vis', help='output directory')
    parser.add_argument('--num-samples', type=int, default=20, help='number of samples to check')
    parser.add_argument('--device', default='cuda', help='device to run augmentation')
    args = parser.parse_args()
    return args

def denormalize(tensor, mean, std):
    """Normalize된 Tensor를 이미지(0-255, RGB)로 변환"""
    tensor = tensor.detach().cpu().clone()
    tensor = tensor.permute(1, 2, 0) # (C, H, W) -> (H, W, C)
    
    mean = torch.tensor(mean).view(1, 1, 3)
    std = torch.tensor(std).view(1, 1, 3)
    
    img = tensor * std + mean
    img = img.clamp(0, 255).numpy().astype(np.uint8)
    return img

def main():
    args = parse_args()
    
    print(f"Loading config: {args.config}")
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmpose'))

    print("Building dataset...")
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    
    print("Building Augmentor...")
    # Config에서 batch_augments 부분 추출
    if hasattr(cfg.model, 'data_preprocessor') and \
       'batch_augments' in cfg.model.data_preprocessor:
        aug_cfg = cfg.model.data_preprocessor.batch_augments[0]
        try:
            augmentor = MODELS.build(aug_cfg)
        except Exception as e:
            print(f"Error: {e}")
            return
        augmentor.to(args.device)
        augmentor.train() 
    else:
        print("Config에 batch_augments가 없습니다.")
        return

    mean_vals = cfg.model.data_preprocessor.mean
    std_vals = cfg.model.data_preprocessor.std
    mean_tensor = torch.tensor(mean_vals).view(1, 3, 1, 1).to(args.device)
    std_tensor = torch.tensor(std_vals).view(1, 3, 1, 1).to(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 증강 이름 매핑
    AUG_NAMES = {0: "Identity (None)", 1: "RandConv", 2: "StyleAug", 3: "DeepAug"}

    print(f"Start visualizing {args.num_samples} samples...")
    
    for i in range(min(len(dataset), args.num_samples)):
        item = dataset[i]
        input_tensor = item['inputs'].float().to(args.device)
        input_batch = input_tensor.unsqueeze(0)
        
        # 1. 정규화 (Augmentor 입력 준비)
        normalized_batch = (input_batch - mean_tensor) / std_tensor
        
        # 2. Augmentation 실행
        augmented_batch, _ = augmentor(normalized_batch, None)
        
        # [NEW] 적용된 증강 정보 가져오기
        aug_name = "Unknown"
        if hasattr(augmentor, 'latest_choices'):
            # batch size가 1이므로 첫번째 요소만 가져옴
            idx = augmentor.latest_choices[0].item()
            aug_name = AUG_NAMES.get(idx, f"Unknown({idx})")
        
        # 콘솔 로깅
        print(f"[Sample {i:02d}] Applied: {aug_name}")

        # 3. 시각화 (역정규화)
        img_org = denormalize(normalized_batch[0], mean_vals, std_vals)
        img_aug = denormalize(augmented_batch[0], mean_vals, std_vals)
        
        # 4. 저장
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(img_org)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        axes[1].imshow(img_aug)
        # [NEW] 이미지 타이틀에 적용된 증강 표시
        axes[1].set_title(f"Augmented: {aug_name}")
        axes[1].axis('off')
        
        save_path = osp.join(args.output_dir, f"sample_{i:03d}_{aug_name.split()[0]}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    print(f"Done. Results saved in {args.output_dir}")

if __name__ == '__main__':
    main()