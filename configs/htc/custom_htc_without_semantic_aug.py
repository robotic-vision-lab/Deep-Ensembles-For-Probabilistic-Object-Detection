_base_ = [
    'custom_classes_htc_without_semantic.py',
]

# classes = (
#     'person', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#     'orange', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
#     'mouse', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock')

classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
           'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
           'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
           'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')
print("len(classes):", len(classes))

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=1, norm_type=2))

data_root = 'D:/Datasets/coco/images/'
dataset_type = 'CocoDataset'
checkpoint_config = dict(create_symlink=False)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='HorizontalFlip',
                p=.1,
            ),
            dict(
                type='RandomGamma',
                p=.1,
            ),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
    ),
    val=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
    ),

    test=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
    ))

"""classes = {'bottle','cup','knife','bowl','wine glass', 'fork','spoon','banana','apple','orange','cake',
'potted plant','mouse','keyboard','laptop','cell phone','book','clock','chair','dining table','couch','bed','toilet',
'tv','microwave','toaster','refrigerator','oven','sink','person'} data = dict( train=dict(classes=classes), 
val=dict(classes=classes), test=dict(classes=classes)) """

load_from = "D:/mmdetection2.0-prod-20210125/mmdetection2.0/checkpoints/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth"
