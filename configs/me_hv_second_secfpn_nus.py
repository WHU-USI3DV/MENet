_base_ = [
    './_base_/datasets/nus_3d_lidar_map.py',
    './_base_/schedules/cyclic_20e.py', 
    './_base_/default_runtime.py'
]

optimizer = dict(type='AdamW', lr=1.25e-4, weight_decay=0.01)

voxel_size = [0.1, 0.1, 0.2]
point_cloud_range_ = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

model = dict(
    type = "BEVFusion",
    encoders = [
        dict(
            type = "VoxelEncoder",
            voxel_layer = dict(
                max_num_points=10,
                point_cloud_range=point_cloud_range_,
                voxel_size=voxel_size,
                max_voxels=(90000, 120000)),
            voxel_encoder = dict(type='HardSimpleVFE', num_features=5),
            middle_encoder = dict(
                type='SparseEncoder',
                in_channels=5,
                sparse_shape=[41, 1024, 1024],
                output_channels=128,
                order=('conv', 'norm', 'act'),
                encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
                encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
                block_type='basicblock'
            ),
            stream_name="lidar"),
        dict(
            type = "ConvMapEncoder", 
            conv_group_cfg = dict(
                type="EdgeResidualGroup",
                input_channels=6,
                out_channels=[32, 64, 64],
                mid_channels=[16, 32, 64],
                strides=[2, 1, 1],
                dilation=3,
                with_se=True, # with SE layer
            ),
            stream_name="map")
    ],
    fuser = dict(
        type = "AttentionFuser",
        in_channels=256+64,
        attention_only=True,
        attention_cfg=dict(type="BAM")
    ),
    decoder = dict(
        type = "SECONDDecoder",
        backbone = dict(
            type='SECOND',
            in_channels=256+64,
            out_channels=[128, 256],
            layer_nums=[5, 5],
            layer_strides=[1, 2],
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
            conv_cfg=dict(type='Conv2d', bias=False)),
        neck = dict(
            type='SECONDFPN',
            in_channels=[128, 256],
            out_channels=[256, 256],
            upsample_strides=[1, 2],
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
            upsample_cfg=dict(type='deconv', bias=False),
            use_conv_for_no_stride=True),
    ),
    
    head = dict(
        type='Anchor3DHead',
        num_classes=10,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[
                [-51.2, -51.2, -1.80032795, 51.2, 51.2, -1.80032795],
                [-51.2, -51.2, -1.74440365, 51.2, 51.2, -1.74440365],
                [-51.2, -51.2, -1.68526504, 51.2, 51.2, -1.68526504],
                [-51.2, -51.2, -1.67339111, 51.2, 51.2, -1.67339111],
                [-51.2, -51.2, -1.61785072, 51.2, 51.2, -1.61785072],
                [-51.2, -51.2, -1.80984986, 51.2, 51.2, -1.80984986],
                [-51.2, -51.2, -1.76396500, 51.2, 51.2, -1.76396500],
            ],
            sizes=[ # lwh
                [4.60718145, 1.95017717, 1.72270761],  # car
                [6.73778078, 2.4560939, 2.73004906],  # truck
                [12.01320693, 2.87427237, 3.81509561],  # trailer
                [1.68452161, 0.60058911, 1.27192197],  # bicycle
                [0.7256437, 0.66344886, 1.75748069],  # pedestrian
                [0.40359262, 0.39694519, 1.06232151],  # traffic_cone
                [0.48578221, 2.49008838, 0.98297065],  # barrier
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=False),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=-0.7854,  # -pi / 4
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.2,
        score_thr=0.05,
        min_bbox_size=0,
        max_num=500)
)
