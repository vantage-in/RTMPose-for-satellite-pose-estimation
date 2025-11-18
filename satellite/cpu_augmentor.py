# my_spn_aug.py

import albumentations as A

#'pixel' 관련 증강
from albumentations.augmentations.pixel.transforms import (
    RandomBrightnessContrast, Posterize, Sharpen,
    Solarize, Spatter, Equalize, RandomGamma, RandomSunFlare, GaussNoise, ISONoise, MultiplicativeNoise
)
# 'blur' 관련 증강
from albumentations.augmentations.blur.transforms import (
    GaussianBlur, MedianBlur, Defocus
)
# 합성(composition) 유틸리티
from albumentations.core.composition import OneOf
from albumentations.core.composition import OneOf
from mmpose.registry import TRANSFORMS
import numpy as np
import warnings

@TRANSFORMS.register_module()
class SPNAugmentation:
    """
    선행 연구의 RandAugment 유사 증강(Table 1)을 적용합니다.
    - N개의 연산을 무작위로 선택하여 순차 적용합니다.
    - RandomSunFlare는 'bbox' 키를 사용하여 GT BBox 내부에만 적용됩니다.
    """
    def __init__(self, n: int = 2, p: float = 1.0):
        self.n = n
        self.p = p
        
        # 1. RandomSunFlare 변환을 별도로 인스턴스화합니다.
        # (p=1.0으로 설정하여, 일단 선택되면 항상 적용되도록 함)
        self.flare_transform = RandomSunFlare(p=1.0, src_radius=100)
        
        # 2. Table 1의 나머지 증강 함수 풀을 정의합니다.
        self.simple_aug_pool = [
            RandomBrightnessContrast(p=1.0),
            OneOf([
                GaussianBlur(blur_limit=(3, 7), p=1.0), 
                MedianBlur(blur_limit=(3, 7), p=1.0), 
                Defocus(radius=(1, 3), p=1.0)
            ], p=1.0),
            OneOf([
                GaussNoise(p=1.0), 
                ISONoise(p=1.0), 
                MultiplicativeNoise(p=1.0)
            ], p=1.0),
            Posterize(num_bits=(4, 8), p=1.0),
            Sharpen(p=1.0),
            Solarize(p=1.0),
            Spatter(p=1.0),
            Equalize(p=1.0),
            RandomGamma(p=1.0),
        ]
        
        # 3. 전체 증강 풀을 결합합니다.
        # (flare_transform을 식별할 수 있도록 리스트에 그대로 추가)
        self.aug_pool = self.simple_aug_pool + [self.flare_transform]

    def _apply_flare_to_bbox(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        RandomSunFlare를 이미지의 BBox 영역에만 적용합니다.
        """
        try:
            if bbox.ndim > 1:
                bbox = bbox[0]
            # BBox 좌표를 정수형으로 변환하고 이미지 경계 내로 클리핑
            x1, y1, x2, y2 = bbox.astype(int)
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x1 >= x2 or y1 >= y2:
                # 유효하지 않은 BBox
                return img

            # BBox 영역을 자릅니다.
            bbox_crop = img[y1:y2, x1:x2]
            
            # 자른 영역에만 SunFlare 증강을 적용합니다.
            augmented_crop = self.flare_transform(image=bbox_crop)['image']
            
            # 원본 이미지를 복사하여 증강된 BBox 영역을 붙여넣습니다.
            img_copy = img.copy()
            img_copy[y1:y2, x1:x2] = augmented_crop
            return img_copy

        except Exception as e:
            warnings.warn(f"BBox SunFlare 적용 실패: {e}. 원본 이미지 반환.")
            return img

    def __call__(self, results: dict) -> dict:
        """MMPose 파이프라인에서 호출됩니다."""
        
        # 1. 전체 증강 블록을 적용할지 확률적으로 결정
        if np.random.rand() > self.p:
            return results
            
        img = results['img']
        
        # 2. BBox 정보 가져오기
        # 'GetBBoxCenterScale' 이후 'bbox' 키가 있어야 함
        if 'bbox' not in results:
            warnings.warn(
                "SPNAugmentation: 'bbox'가 results에 없습니다. "
                "RandomSunFlare가 전체 이미지에 적용될 수 있습니다."
            )
            bbox = None
        else:
            # `results['bbox']`는 [x1, y1, w, h] 형식일 수 있습니다.
            # `albumentations`는 [x1, y1, x2, y2] 형식이 필요합니다.
            # `GetBBoxCenterScale`은 'bbox'를 [x1, y1, x2, y2]로 저장합니다.
            bbox = results['bbox'] 

        # 3. N개의 증강을 무작위로 선택
        chosen_indices = np.random.choice(
            len(self.aug_pool), self.n, replace=False
        )
        
        # 4. 선택된 증강을 순차적으로 적용
        for idx in chosen_indices:
            aug_fn = self.aug_pool[idx]
            
            # 5. 선택된 증강이 RandomSunFlare인지 확인
            if aug_fn is self.flare_transform:
                if bbox is not None:
                    # BBox가 있으면 BBox 내부에만 적용
                    img = self._apply_flare_to_bbox(img, bbox)
                else:
                    # BBox가 없으면 (경고 후) 전체 이미지에 적용
                    img = aug_fn(image=img)['image']
            else:
                # 다른 모든 "간단한" 증강은 전체 이미지에 적용
                img = aug_fn(image=img)['image']
        
        results['img'] = img
        return results