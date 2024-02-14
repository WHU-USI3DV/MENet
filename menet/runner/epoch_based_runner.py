from mmcv.runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS

@RUNNERS.register_module(name="EpochBasedRunner", force=True)
class EpochBasedRunnerStopEpoch(EpochBasedRunner):
    def train(self, data_loader, **kwargs):
        # set transform's epoch
        dataset_tmp = data_loader.dataset
        if dataset_tmp.__class__.__name__ == "CBGSDataset":
            dataset_tmp = dataset_tmp.dataset

        if hasattr(dataset_tmp, "pipeline"):
            for transform in dataset_tmp.pipeline.transforms:
                if hasattr(transform, "set_epoch"):
                    transform.set_epoch(self.epoch)
        super().train(data_loader, **kwargs)