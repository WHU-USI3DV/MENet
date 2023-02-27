_base_ = [
    './_base_/datasets/nus_3d_lidar_map.py',
    './_base_/models/menet.py',
    './_base_/schedules/cyclic_20e.py', 
    './_base_/default_runtime.py'
]

# cycle lr = 1e-4, BatchSize=4
# schedulex2 lr = 0.001, bs = 4
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)

workflow = [('train', 2), ('val', 1)]
