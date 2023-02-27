# NuScenes Dateset for CenterHead

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range_d = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

map_classes = [
    'drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area',
    'divider'
]

# input_modality = dict(
#     use_lidar=True,
#     use_camera=False,
#     use_radar=False,
#     use_map=False,
#     use_external=False)

xbound = [-51.2, 51.2, 0.4]
ybound = [-51.2, 51.2, 0.4]

dataset_type = 'NuScenesDataset'
# data_root = 'data/nuscenes/v1.0-mini/'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

db_sampler = dict(
    type = "MapEnhancedDataBaseSampler",
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    xbound = xbound,
    ybound = ybound,
    map_classes = map_classes,
    bbox_code_size = None,
    classes = class_names,
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='LoadMapMask',
        data_root=data_root,
        xbound=xbound,
        ybound=ybound,
        classes=map_classes),
    dict(type='ObjectSample', stop_epoch=10, db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range_d),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range_d),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='CollectFusion', 
        input_data_keys=("points", "map_mask"),
        gt_keys=('gt_bboxes_3d', 'gt_labels_3d'))
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='LoadMapMask',
        data_root=data_root,
        xbound=xbound,
        ybound=ybound,
        classes=map_classes),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='CollectFusion', 
        input_data_keys=("points", "map_mask"))
]

# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = test_pipeline

load_interval = 1
with_velocity = True

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        map_classes=map_classes,
        load_interval=load_interval,
        test_mode=False,
        with_velocity=with_velocity,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'
        ),
    val=dict(
        type=dataset_type,
        # data_root=data_root + '/v1.0-mini/',
        # ann_file=data_root + '/v1.0-mini/' + 'nuscenes_infos_val.pkl',
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        map_classes=map_classes,
        # modality=input_modality,
        with_velocity=with_velocity,
        test_mode=True,
        box_type_3d='LiDAR'
        ),
    test=dict(
        type=dataset_type,
        # data_root=data_root + '/v1.0-mini/',
        # ann_file=data_root + '/v1.0-mini/' + 'nuscenes_infos_val.pkl',
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        map_classes=map_classes,
        # modality=input_modality,
        with_velocity=with_velocity,
        test_mode=True,
        box_type_3d='LiDAR'
    ))

evaluation = dict(interval=20, pipeline=eval_pipeline)