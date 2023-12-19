model = dict(
    detector=dict(
        type='FasterRCNN',
        backbone=dict(
            type='ResNeXt',
            depth=101,
            num_stages=4,
            out_indices=(3, ),
            strides=(1, 2, 2, 1),
            dilations=(1, 1, 1, 2),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d'),
            groups=64,
            base_width=4),
        neck=dict(
            type='ChannelMapper',
            in_channels=[2048],
            out_channels=512,
            kernel_size=3),
        rpn_head=dict(
            type='RPNHead',
            in_channels=512,
            feat_channels=512,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[4, 8, 16, 32],
                ratios=[0.5, 1.0, 2.0],
                strides=[16]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss', beta=0.1111111111111111,
                loss_weight=1.0)),
        roi_head=dict(
            type='SelsaRoIHead',
            bbox_roi_extractor=dict(
                type='TemporalRoIAlign',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=512,
                featmap_strides=[16],
                num_most_similar_points=2,
                num_temporal_attention_blocks=4),
            bbox_head=dict(
                type='SelsaBBoxHead',
                in_channels=512,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.2, 0.2, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                num_shared_fcs=3,
                aggregator=dict(
                    type='SelsaAggregator',
                    in_channels=1024,
                    num_attention_blocks=16))),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=0,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=6000,
                max_per_img=600,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            rpn=dict(
                nms_pre=6000,
                max_per_img=300,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0001,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100))),
    type='SELSA')
dataset_type = 'CocoVideoDataset'
data_root = ''
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqResize', img_scale=(1000, 1000), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(
        type='SeqNormalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqResize', img_scale=(1000, 1000), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
    dict(
        type='SeqNormalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img'],
        meta_keys=('num_left_ref_imgs', 'frame_stride')),
    dict(type='ConcatVideoReferences'),
    dict(type='MultiImagesToTensor', ref_prefix='ref'),
    dict(type='ToList')
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=[
        dict(
            type='CocoVideoDataset',
            classes=('can', ),
            ann_file='orig/cocovid_train_annotations.json',
            img_prefix=None,
            ref_img_sampler=dict(
                num_ref_imgs=1,
                frame_range=1,
                filter_key_img=False,
                method='uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile'),
                dict(
                    type='SeqLoadAnnotations', with_bbox=True,
                    with_track=True),
                dict(type='SeqResize', img_scale=(1000, 1000), keep_ratio=True),
                dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
                dict(
                    type='SeqNormalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='SeqPad', size_divisor=16),
                dict(
                    type='VideoCollect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
                dict(type='ConcatVideoReferences'),
                dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
            ])
    ],
    val=dict(
        type='CocoVideoDataset',
        ann_file='orig/cocovid_val_annotations.json',
        img_prefix=None,
        ref_img_sampler=dict(
            num_ref_imgs=1,
            frame_range=1,
            filter_key_img=False,
            method='uniform'),
        pipeline=[
            dict(type='LoadMultiImagesFromFile'),
            dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
            dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
            dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
            dict(
                type='SeqNormalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='SeqPad', size_divisor=16),
            dict(
                type='VideoCollect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
            dict(type='ConcatVideoReferences'),
            dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
        ],
        test_mode=False,
        classes=('can', )),
    test=dict(
        type='CocoVideoDataset',
        ann_file='orig/cocovid_train_annotations.json',
        img_prefix=None,
        ref_img_sampler=dict(
            num_ref_imgs=1,
            frame_range=1,
            filter_key_img=False,
            method='uniform'),
        pipeline=[
            dict(type='LoadMultiImagesFromFile'),
            dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
            dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
            dict(
                type='SeqNormalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='SeqPad', size_divisor=16),
            dict(
                type='VideoCollect',
                keys=['img'],
                meta_keys=('num_left_ref_imgs', 'frame_stride')),
            dict(type='ConcatVideoReferences'),
            dict(type='MultiImagesToTensor', ref_prefix='ref'),
            dict(type='ToList')
        ],
        test_mode=True,
        classes=('can', )))
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'pretrained/selsa_troialign_faster_rcnn_x101_dc5_7e_imagenetvid_20210822_164036-4471ac42.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[2, 5])
total_epochs = 7
evaluation = dict(metric=['bbox'], interval=7)
classes = ('can',)
work_dir = './work_dirs/custom'
gpu_ids = [0]
