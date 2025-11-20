# tensor_augmentor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from typing import List, Optional, Tuple

from mmpose.registry import MODELS
from mmpose.structures import PoseDataSample as DataSample

# ----------------------------------------------------------------------
# 의존성 임포트
# ----------------------------------------------------------------------
try:
    from styleaug import StyleAugmentor
except ImportError:
    print("Warning: styleaug not installed.")
    StyleAugmentor = None # Handle missing library gracefully if needed

try:
    # CAE_Model 폴더가 PYTHONPATH에 있거나 현재 경로에 있어야 함
    from CAE_Model.cae_32x32x32_zero_pad_bin import CAE
except ImportError:
    # 경로 문제시 예외 처리 혹은 경로 추가 로직 필요
    CAE = None 

# ----------------------------------------------------------------------
# 1. RandConv 구현
# ----------------------------------------------------------------------
class _RandConvImpl(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)

    def forward(self, x):
        weight = torch.randn_like(self.weight)
        weight = F.normalize(weight, dim=[1, 2, 3], p=2)
        if self.bias is not None:
            bias = torch.randn_like(self.bias)
            bias = F.normalize(bias, dim=0, p=2)
        else:
            bias = None
        return F.conv2d(x, weight, bias, self.stride, self.padding,
                        self.dilation, self.groups)


# ----------------------------------------------------------------------
# 2. CAE Wrappers (해결책)
# CAE 클래스가 encoder/decoder 속성이 없으므로, 레이어를 직접 호출하는 래퍼 정의
# ----------------------------------------------------------------------
class CAEEncoderWrapper(nn.Module):
    def __init__(self, cae_model):
        super().__init__()
        self.cae = cae_model

    def forward(self, x):
        # CAE.forward의 앞부분(인코딩) 로직을 그대로 가져옴
        ec1 = self.cae.e_conv_1(x)
        ec2 = self.cae.e_conv_2(ec1)
        eblock1 = self.cae.e_block_1(ec2) + ec2
        eblock2 = self.cae.e_block_2(eblock1) + eblock1
        eblock3 = self.cae.e_block_3(eblock2) + eblock2
        ec3 = self.cae.e_conv_3(eblock3)  # Tanh activation -> range [-1, 1]
        return ec3

class CAEDecoderWrapper(nn.Module):
    def __init__(self, cae_model):
        super().__init__()
        self.cae = cae_model

    def forward(self, z):
        # CAE.decode 메서드는 입력이 (0,1) 범위라고 가정하고 *2-1을 수행함.
        # 하지만 우리는 EncoderWrapper에서 Tanh 출력(-1~1)을 바로 가져오고
        # 거기에 노이즈를 더할 것이므로, *2-1 변환 없이 바로 레이어에 넣어야 함.
        
        # y = z # z is already expected to be in range approx [-1, 1]
        
        uc1 = self.cae.d_up_conv_1(z)
        dblock1 = self.cae.d_block_1(uc1) + uc1
        dblock2 = self.cae.d_block_2(dblock1) + dblock1
        dblock3 = self.cae.d_block_3(dblock2) + dblock2
        uc2 = self.cae.d_up_conv_2(dblock3)
        dec = self.cae.d_up_conv_3(uc2)
        return dec


# ----------------------------------------------------------------------
# 3. 통합 증강 래퍼
# ----------------------------------------------------------------------
@MODELS.register_module()
class CombinedAugmentation(nn.Module):
    """
    4가지 케이스 중 하나를 랜덤하게 선택하여 적용 (각 확률 1/4)
    0: Identity (적용 안 함)
    1: RandConv
    2: StyleAugment
    3: DeepAugment (CAE)
    """
    def __init__(self,
                 mean: List[float] = [123.675, 116.28, 103.53],
                 std: List[float] = [58.395, 57.12, 57.375],
                 # prob_ 인자들은 호환성을 위해 남겨두지만, 내부 로직에서는 1/4로 고정됨
                 prob_style: float = 0.25, 
                 prob_deep: float = 0.25,
                 prob_randconv: float = 0.25,
                 cae_weights_path: str = '/root/RTMPose/satellite/CAE_Weight/model_final.state',
                 deepaug_sigma: float = 0.1,
                 randconv_kernel_size: int = 3):
        super().__init__()
        
        self.deepaug_sigma = deepaug_sigma

        # 1. 정규화/역정규화용 파라미터
        self.mean = nn.Parameter(
            torch.tensor(mean).view(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(
            torch.tensor(std).view(1, 3, 1, 1), requires_grad=False)

        # 2. StyleAugmentor 초기화
        if StyleAugmentor is not None:
            self.style_augmentor = StyleAugmentor()
        else:
            self.style_augmentor = None

        # 3. DeepAugment (CAE) 초기화
        print(f"Loading CAE weights from: {cae_weights_path}")
        if CAE is None:
            raise ImportError("CAE class not found. Check imports.")
        
        self.cae_encoder, self.cae_decoder = self._load_cae_model(cae_weights_path)

        # 4. RandConv 초기화
        self.rand_conv = _RandConvImpl(
            in_channels=3,
            out_channels=3,
            kernel_size=randconv_kernel_size,
            stride=1,
            padding=randconv_kernel_size // 2,
            dilation=1,
            groups=3,
            bias=False,
            padding_mode='reflect')

    def _load_cae_model(self, weights_path):
        try:
            cae_model = CAE()
            state_dict = torch.load(weights_path, map_location='cpu')
            
            if 'model_state' in state_dict:
                cae_model.load_state_dict(state_dict['model_state'])
            else:
                cae_model.load_state_dict(state_dict)
                
            cae_model.eval()
            for param in cae_model.parameters():
                param.requires_grad = False

            print("DeepAugment (CAE) model loaded successfully.")
            
            encoder = CAEEncoderWrapper(cae_model)
            decoder = CAEDecoderWrapper(cae_model)
            return encoder, decoder
            
        except Exception as e:
            raise IOError(f"CAE 모델 로드 오류: {e}")

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.style_augmentor:
            self.style_augmentor.to(*args, **kwargs)
        self.cae_encoder.to(*args, **kwargs)
        self.cae_decoder.to(*args, **kwargs)
        self.rand_conv.to(*args, **kwargs)
        return self

    def forward(
        self, inputs: torch.Tensor, data_samples: Optional[List[DataSample]]
    ) -> Tuple[torch.Tensor, Optional[List[DataSample]]]:

        # 훈련 모드에서만 증강 적용
        if not self.training:
            return inputs, data_samples

        device = inputs.device
        B = inputs.shape[0]

        # --- 1. 역정규화 (Mean/Std -> 0~1) ---
        inputs_unnormalized = inputs * self.std + self.mean
        current_tensor = (inputs_unnormalized / 255.0).clamp(0.0, 1.0)

        # --- 2. 배타적 선택 로직 (Mutually Exclusive Selection) ---
        # 0: None, 1: RandConv, 2: StyleAug, 3: DeepAug (각 25%)
        choices = torch.randint(0, 4, (B,), device=device)
        self.latest_choices = choices

        # CASE 1: RandConv 적용 (Choice == 1)
        # 해당 인덱스만 추출하여 연산하므로 효율적임
        idx_rand = (choices == 1).nonzero(as_tuple=True)[0]
        if len(idx_rand) > 0:
            sub_inputs = current_tensor[idx_rand]
            rand_out = self.rand_conv(sub_inputs)
            # [수정] Alpha Blending 적용
            # alpha가 클수록 RandConv 효과가 강함 (0.0 ~ 1.0)
            alpha = torch.rand(len(idx_rand), 1, 1, 1, device=device) * 0.1 + 0.8  # 0.8 ~ 0.9 사이 랜덤
            
            # 원본과 RandConv 결과를 섞음
            mixed_out = (1 - alpha) * sub_inputs + alpha * rand_out
            
            current_tensor[idx_rand] = mixed_out

        # CASE 2: StyleAug 적용 (Choice == 2)
        idx_style = (choices == 2).nonzero(as_tuple=True)[0]
        if len(idx_style) > 0 and self.style_augmentor is not None:
            sub_inputs = current_tensor[idx_style]
            style_out = self.style_augmentor(sub_inputs)
            current_tensor[idx_style] = style_out

        # CASE 3: DeepAug (CAE) 적용 (Choice == 3)
        idx_deep = (choices == 3).nonzero(as_tuple=True)[0]
        if len(idx_deep) > 0:
            sub_inputs = current_tensor[idx_deep]
            # 인코더 -> 노이즈 -> 디코더
            z = self.cae_encoder(sub_inputs)
            delta = torch.randn_like(z) * self.deepaug_sigma
            z_perturbed = z + delta
            deep_out = self.cae_decoder(z_perturbed)
            current_tensor[idx_deep] = deep_out

        # CASE 0: None (Choice == 0)
        # 아무것도 하지 않으므로 코드 작성 불필요 (이미 current_tensor에 원본 유지됨)

        # --- 3. 다시 정규화 (0~1 -> Mean/Std) ---
        current_tensor = (current_tensor * 255.0 - self.mean) / self.std

        return current_tensor, data_samples