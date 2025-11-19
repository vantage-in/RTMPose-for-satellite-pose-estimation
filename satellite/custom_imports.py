custom_imports = dict(
    imports=['cpu_augmentor', 'tensor_augmentor'], # 리스트 안에 쉼표로 구분하여 나열
    allow_failed_imports=False
)
transforms = dict(type='SPNAugmentation', n=2, p=0.8)
model = dict(type='CombinedAugmentation')