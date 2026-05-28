from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.nn as nn
from plugin.utils.fp16_utils import auto_fp16
from mmengine.logging import print_log
from mmengine.utils import is_list_of

from mmengine.logging import MMLogger
from mmdet3d.registry import MODELS as MMDET3D_MODELS

MAPPERS = MMDET3D_MODELS

class BaseMapper(nn.Module, metaclass=ABCMeta):
    """Base class for mappers."""

    def __init__(self):
        super(BaseMapper, self).__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
                or (hasattr(self, 'mask_head') and self.mask_head is not None))

    #@abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass

    def forward_train(self, *args, **kwargs):
        pass

    #@abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    #@abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if pretrained is not None:
            logger = MMLogger.get_current_instance()
            print_log(f'load model from: {pretrained}', logger=logger)

    def forward_test(self, *args, **kwargs):
        """
        Args:
        """
        if True:
            self.simple_test()
        else:
            self.aug_test()

    # @auto_fp16(apply_to=('img', ))
    def forward(self, *args, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(*args, **kwargs)
        else:
            kwargs.pop('rescale')
            return self.forward_test(*args, **kwargs)

    def train_step(self, data_dict, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_dict (dict): The output of dataloader.
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
        loss, log_vars, num_samples = self(**data_dict)
        
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=num_samples)

        return outputs

    def parse_losses(self, losses):
        """Parse the raw outputs (losses) from the network."""

        if isinstance(losses, tuple):
            assert len(losses) == 3
            loss, log_vars, _ = losses
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, dtype=torch.float32, device=next(self.parameters()).device)
            log_vars_tensor = OrderedDict()
            for name, value in log_vars.items():
                if isinstance(value, torch.Tensor):
                    log_vars_tensor[name] = value.to(loss.device)
                else:
                    log_vars_tensor[name] = torch.tensor(
                        value, dtype=torch.float32, device=loss.device)
            log_vars_tensor['loss'] = loss
            return loss, log_vars_tensor

        if isinstance(losses, dict):
            log_vars = []
            for loss_name, loss_value in losses.items():
                if isinstance(loss_value, torch.Tensor):
                    log_vars.append([loss_name, loss_value.mean()])
                elif is_list_of(loss_value, torch.Tensor):
                    log_vars.append(
                        [loss_name,
                         sum(_loss.mean() for _loss in loss_value)])
                else:
                    raise TypeError(
                        f'{loss_name} is not a tensor or list of tensors')

            loss = sum(value for key, value in log_vars if 'loss' in key)
            log_vars.insert(0, ['loss', loss])
            log_vars = OrderedDict(log_vars)
            return loss, log_vars

        raise TypeError('Unsupported losses type for parsing.')

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        loss, log_vars, num_samples = self(**data)
        
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=num_samples)

        return outputs

    def show_result(self,
                    **kwargs):
        img = None
        return img
