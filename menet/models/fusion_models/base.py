from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict

import torch
import torch.distributed as dist
from mmcv.runner import BaseModule
from mmcv.runner import auto_fp16
from mmcv.utils.logging import get_logger, logger_initialized, print_log

__all__ = ["Base3DFusionModel"]

class Base3DFusionModel(BaseModule, metaclass=ABCMeta):
    """Base class for fusion_models."""

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.fp16_enabled = False

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

        log_vars["loss"] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data["metas"]))
        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data["metas"]))

        return outputs

    @auto_fp16(apply_to=('input_data'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_test(self, input_data, metas, rescale=False, **kwargs):
        """
        TODO: surpport test time augmentation.
        """

        for var, name in [(input_data, 'input_data')]:
            assert not isinstance(var, list), \
                'TTA do not support. {} must not be a list, but got {}'.format(name, type(var))
        return self.simple_test(input_data, metas, rescale=False, **kwargs)

    @abstractmethod
    def simple_test(self, **kwargs):
        pass

    @abstractmethod
    def forward_train(self, **kwargs):
        pass

    def init_weights(self):
        if self.init_cfg is None:
            super().init_weights()
        else: # only initialize by the root model
            is_top_level_module = False
            # check if it is top-level module
            if not hasattr(self, '_params_init_info'):
                # The `_params_init_info` is used to record the initialization
                # information of the parameters
                # the key should be the obj:`nn.Parameter` of model and the value
                # should be a dict containing
                # - init_info (str): The string that describes the initialization.
                # - tmp_mean_value (FloatTensor): The mean of the parameter,
                #       which indicates whether the parameter has been modified.
                # this attribute would be deleted after all parameters
                # is initialized.
                self._params_init_info: defaultdict = defaultdict(dict)
                is_top_level_module = True

                # Initialize the `_params_init_info`,
                # When detecting the `tmp_mean_value` of
                # the corresponding parameter is changed, update related
                # initialization information
                for name, param in self.named_parameters():
                    self._params_init_info[param][
                        'init_info'] = f'The value is the same before and ' \
                                    f'after calling `init_weights` ' \
                                    f'of {self.__class__.__name__} '
                    self._params_init_info[param][
                        'tmp_mean_value'] = param.data.mean()

                # pass `params_init_info` to all submodules
                # All submodules share the same `params_init_info`,
                # so it will be updated when parameters are
                # modified at any level of the model.
                for sub_module in self.modules():
                    sub_module._params_init_info = self._params_init_info

            # Get the initialized logger, if not exist,
            # create a logger named `mmcv`
            logger_names = list(logger_initialized.keys())
            logger_name = logger_names[0] if logger_names else 'mmcv'

            from mmcv.cnn import initialize
            from mmcv.cnn.utils.weight_init import update_init_info
            module_name = self.__class__.__name__
            if not self._is_init:
                if self.init_cfg:
                    print_log(
                        f'initialize {module_name} with init_cfg {self.init_cfg}',
                        logger=logger_name)
                    initialize(self, self.init_cfg)
                    if isinstance(self.init_cfg, dict):
                        # prevent the parameters of
                        # the pre-trained model
                        # from being overwritten by
                        # the `init_weights`
                        if self.init_cfg['type'] == 'Pretrained':
                            return
                self._is_init = True
            else:
                warnings.warn(f'init_weights of {self.__class__.__name__} has '
                            f'been called more than once.')

            if is_top_level_module:
                self._dump_init_info(logger_name)

                for sub_module in self.modules():
                    del sub_module._params_init_info