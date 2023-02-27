_base_ = [
    './_base_/models/hv_second_secfpn_nus.py',
    './_base_/datasets/nus-3d_lidar.py', 
    './_base_/schedules/cyclic_20e.py',
    './_base_/default_runtime.py'
]

optimizer = dict(type='AdamW', lr=1.25e-4, weight_decay=0.01)