backend_args = dict(backend='local')
base_lr = 0.001
codec = dict(
    input_size=(
        224,
        224,
    ),
    normalize=False,
    sigma=(
        4.9,
        5.66,
    ),
    simcc_split_ratio=2.0,
    type='SimCCLabel',
    use_dark=False)
custom_hooks = [
    dict(_scope_='mmpose', type='SyncBuffersHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'cpu_augmentor',
        'tensor_augmentor',
        'bbox',
    ])
data_mode = 'topdown'
data_root = '/workspace/speedplusv2/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    badcase=dict(
        _scope_='mmpose',
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(_scope_='mmpose', interval=10, type='CheckpointHook'),
    logger=dict(_scope_='mmpose', interval=50, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmpose', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmpose', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmpose', type='IterTimerHook'),
    visualization=dict(
        _scope_='mmpose', enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
input_size = (
    224,
    224,
)
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(
    _scope_='mmpose',
    by_epoch=True,
    num_digits=6,
    type='LogProcessor',
    window_size=50)
max_epochs = 30
model = dict(
    backbone=dict(
        _scope_='mmdet',
        act_cfg=dict(type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.67,
        expand_ratio=0.5,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth',
            prefix='backbone.',
            type='Pretrained'),
        norm_cfg=dict(type='SyncBN'),
        out_indices=(4, ),
        type='CSPNeXt',
        widen_factor=0.75),
    data_preprocessor=dict(
        batch_augments=[
            dict(
                cae_weights_path=
                '/root/RTMPose/satellite/CAE_Weight/model_final.state',
                deepaug_sigma=0.1,
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                prob_deep=0.5,
                prob_randconv=0.5,
                prob_style=0.5,
                randconv_kernel_size=3,
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                type='CombinedAugmentation'),
        ],
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            input_size=(
                224,
                224,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        final_layer_kernel_size=7,
        gau_cfg=dict(
            act_fn='SiLU',
            drop_path=0.0,
            dropout_rate=0.0,
            expansion_factor=2,
            hidden_dims=256,
            pos_enc=False,
            s=128,
            use_rel_bias=False),
        in_channels=768,
        in_featuremap_size=(
            7,
            7,
        ),
        input_size=(
            224,
            224,
        ),
        loss=dict(
            beta=10.0,
            label_softmax=True,
            type='KLDiscretLoss',
            use_target_weight=True),
        out_channels=11,
        simcc_split_ratio=2.0,
        type='RTMCCHead'),
    test_cfg=dict(flip_test=True),
    type='TopdownPoseEstimator')
num_keypoints = 11
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=1,
        start_factor=0.001,
        type='LinearLR'),
    dict(
        T_max=15,
        begin=15,
        by_epoch=True,
        convert_to_iter_based=True,
        end=30,
        eta_min=5e-05,
        type='CosineAnnealingLR'),
]
randomness = dict(seed=42)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotations/validation.json',
        data_mode='topdown',
        data_prefix=dict(img='val/'),
        data_root='/workspace/speedplusv2/',
        metainfo=dict(from_file='configs/_base_/datasets/custom.py'),
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='SetFullImageBBox'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                224,
                224,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/workspace/speedplusv2/annotations/validation.json',
    type='CocoMetric')
train_batch_size = 32
train_cfg = dict(by_epoch=True, max_epochs=30)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotations/train.json',
        data_mode='topdown',
        data_prefix=dict(img='train/'),
        data_root='/workspace/speedplusv2/',
        metainfo=dict(from_file='configs/_base_/datasets/custom.py'),
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='SetFullImageBBox'),
            dict(type='GetBBoxCenterScale'),
            dict(n=2, p=0.8, type='SPNAugmentation'),
            dict(input_size=(
                224,
                224,
            ), type='TopdownAffine'),
            dict(
                encoder=dict(
                    input_size=(
                        224,
                        224,
                    ),
                    normalize=False,
                    sigma=(
                        4.9,
                        5.66,
                    ),
                    simcc_split_ratio=2.0,
                    type='SimCCLabel',
                    use_dark=False),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='CocoDataset'),
    num_workers=10,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='SetFullImageBBox'),
    dict(type='GetBBoxCenterScale'),
    dict(n=2, p=0.8, type='SPNAugmentation'),
    dict(input_size=(
        224,
        224,
    ), type='TopdownAffine'),
    dict(
        encoder=dict(
            input_size=(
                224,
                224,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
transforms = dict(n=2, p=0.8, type='SPNAugmentation')
transforms2 = dict(type='SetFullImageBBox')
val_batch_size = 32
val_cfg = None
val_dataloader = None
val_evaluator = None
val_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='SetFullImageBBox'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        224,
        224,
    ), type='TopdownAffine'),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(_scope_='mmpose', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmpose',
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/rtmpose'
