# satellite/tensor_augmentor.py 파일 하단이나 custom_imports.py에 추가
import numpy as np
from mmpose.registry import TRANSFORMS

@TRANSFORMS.register_module()
class SetFullImageBBox:
    """
    이미지가 이미 Cropped/Resized 된 상태일 때,
    이미지 전체 영역을 BBox로 설정하여 GetBBoxCenterScale이 작동하도록 함.
    """
    def __init__(self):
        pass

    def __call__(self, results):
        img_shape = results['img_shape'] # (h, w)
        h, w = img_shape[:2]
        
        # [x1, y1, x2, y2] = [0, 0, w, h]
        # 점수는 1.0으로 설정
        results['bbox'] = np.array([[0, 0, w, h]], dtype=np.float32)
        results['bbox_score'] = np.ones(1, dtype=np.float32)
        
        return results