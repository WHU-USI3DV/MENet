from mmcv.runner import BaseModule

class BaseEncoder(BaseModule):
    def __init__(self, stream_name="base", init_cfg=None):
        super(BaseEncoder, self).__init__(init_cfg)
        self.stream_name = stream_name

    def loss(self, input_data, gt_labels, metas=None):
        pass