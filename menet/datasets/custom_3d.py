from mmdet3d.datasets import Custom3DDataset, DATASETS

@DATASETS.register_module(name="Custom3DDataset", force=True)
class Custom3DDatasetCustom(Custom3DDataset):
    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or ("gt_labels" in example.keys() and \
                    ~(example['gt_labels']['gt_labels_3d']._data != -1).any())):
            return None
        return example