_base_ = ['mmpose::_base_/default_runtime.py']

import_from = [
    'satellite.cpu_augmentor',
    'satellite.tensor_augmentor'
]

from mmengine.config import Config
cfg = Config.fromfile('satellite/custom_imports.py')
from mmpose.registry import TRANSFORMS
aug = TRANSFORMS.build(cfg.transforms)
print(aug)


# custom_imports = dict(imports=['my_module'], allow_failed_imports=False)

# common setting
num_keypoints = 11
input_size = (224, 224)

# runtime
max_epochs = 30
base_lr = 1e-3
train_batch_size = 32
val_batch_size = 32

train_cfg = dict(max_epochs=max_epochs, val_interval=30)
randomness = dict(seed=42)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=1e-4),
    clip_grad=dict(max_norm=35, norm_type=2), 
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=True, 
        begin=0,
        end=1),
    dict(
        type='CosineAnnealingLR', # 절반부터 시작
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        # --- 통합 증강 프레임워크 적용 ---
        batch_augments=[
            dict(
                type='CombinedAugmentation',
                # rtmpose.py의 값과 동일하게 설정
                mean=[123.675, 116.28, 103.53], 
                std=[58.395, 57.12, 57.375],      
                
                # 각 증강의 적용 확률 (필요에 따라 조절)
                prob_style=0.5,
                prob_deep=0.5,
                prob_randconv=0.5,
                
                # DeepAugment(CAE) 설정
                cae_weights_path='CAE_Weight/model_final.state', #
                deepaug_sigma=0.1, # 논문[25]에서 사용한 노이즈 강도
                
                # RandConv 설정
                randconv_kernel_size=3
            )
        ]),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt', # the ImageNet classification pre-trained weights of the CSPNeXt backbone 
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth'  # noqa 
        )),
    head=dict( 
        type='RTMCCHead',
        in_channels=768,
        out_channels=num_keypoints,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '/workspace/speedplusv2'

backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='SPNAugmentation', n=2, p=0.8), # n=논문에서 사용한 N값, p=적용 확률
    dict(type='GenerateTarget', encoder=codec), # label 변환
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=10, # cpu 코어 수
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline,
        metainfo=dict(from_file='configs/_base_/datasets/custom.py'),
    ))
val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/validation.json',
        # bbox_file=f'{data_root}person_detection_results/'
        # 'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dict(from_file='configs/_base_/datasets/custom.py'),
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/validation.json')
test_evaluator = val_evaluator