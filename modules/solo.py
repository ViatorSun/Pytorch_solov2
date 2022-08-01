#  !/usr/bin/env  python
#  -*- coding:utf-8 -*-
# @Time     :  2022.07
# @Author   :  绿色羽毛
# @Email    :  lvseyumao@foxmail.com
# @Blog     :  https://blog.csdn.net/ViatorSun
# @arXiv    :   
# @version  :   
# @Note     :   
#
#


# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings

import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core.visualization.image import imshow_det_bboxes
# from ..builder import DETECTORS
# from base_mask_head import BaseMaskHead
from .nninit import xavier_init
from .backbone import resnet18, resnet34
from .solo_head import SOLOHead
from .nninit import xavier_init, kaiming_init, normal_init


INF = 1e8


class FPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 upsample_cfg=dict(mode='nearest')):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels    = in_channels  # [64, 128, 256, 512] resnet18
        self.out_channels   = out_channels  # 256
        self.num_ins        = len(in_channels)  # 4
        self.num_outs       = num_outs  # 5
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg   = upsample_cfg.copy()
        if end_level == -1:  # default -1
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level

        self.start_level = start_level
        self.end_level   = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2d(in_channels[i], out_channels, kernel_size=1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:  # default false
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]

        used_backbone_levels = len(laterals)

        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')

        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


class SingleStageInstanceSegmentor(nn.Module):
    """Base class for single-stage instance segmentors."""

    def __init__(self,
                 backbone=None,
                 neck=FPN(in_channels=[64, 128, 256, 512],out_channels=256, num_outs=5),
                 bbox_head=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 mode = 'train',
                 init_cfg=None):

        super(SingleStageInstanceSegmentor, self).__init__()

        # if backbone.name == 'resnet18':
        # if True:
        # self.backbone = resnet18(pretrained=True, loadpath='./pretrained/resnet18_nofc.pth')
        self.backbone = resnet34(pretrained=True, loadpath='./pretrained/resnet34_nofc.pth')
        # elif backbone.name == 'resnet34':
        #     self.backbone = resnet34(pretrained=True, loadpath='./pretrained/resnet34_nofc.pth')

        if neck is not None:
            self.with_neck = True
            self.neck = FPN(in_channels=[64, 128, 256, 512],out_channels=256, num_outs=5,
                            start_level=0,upsample_cfg=dict(mode='nearest'))
        else:
            self.neck = None

        if bbox_head is not None:
            bbox_head.update(train_cfg=copy.deepcopy(train_cfg))
            bbox_head.update(test_cfg=copy.deepcopy(test_cfg))
            # self.bbox_head = build_head(bbox_head)
        else:
            self.bbox_head = None

        # assert mask_head, f'`mask_head` must be implemented in {self.__class__.__name__}'
        # mask_head.update(train_cfg=copy.deepcopy(train_cfg))
        # mask_head.update(test_cfg=copy.deepcopy(test_cfg))
        # {'nms_pre': 500, 'score_thr': 0.1, 'mask_thr': 0.5, 'filter_thr': 0.05,
        #  'kernel': 'gaussian', 'sigma': 2.0,'max_per_img': 100}
        self.mask_head = SOLOHead( num_classes=28, in_channels=256, stacked_convs=7,
                                   feat_channels=256, strides=[8, 8, 16, 32, 32],
                                   scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
                                   pos_scale=0.2, num_grids=[40, 36, 24, 16, 12], cls_down_index=0,
                                   norm_cfg={'type': 'GN', 'num_groups': 32, 'requires_grad': True})

        self.test_cfg = test_cfg
        self.mode = mode
        if self.mode == 'train':
            self.backbone.train(mode=True)
        else:
            self.backbone.train(mode=True)

        if pretrained is None:
            self.init_weights()             # if first train, use this initweight
        else:
            self.load_weights(pretrained)   # load weight from file

    def init_weights(self):
        # fpn
        if isinstance(self.neck, nn.Sequential):
            for m in self.neck:
                m.init_weights()
        else:
            self.neck.init_weights()

        # mask feature mask
        if isinstance(self.mask_head, nn.Sequential):
            for m in self.mask_head:
                m.init_weights()
        else:
            self.mask_head.init_weights()

        # self.bbox_head.init_weights()


    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)


    def load_weights(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)


    def extract_feat(self, img):
        """Directly extract features from the backbone and neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x


    def forward_dummy(self, img):
        """Used for computing network flops.
        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        raise NotImplementedError(f'`forward_dummy` is not implemented in {self.__class__.__name__}')


    def forward(self, img, img_meta, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)


    def forward_train(self, img, img_metas, gt_masks, gt_labels,
                      gt_bboxes=None, gt_bboxes_ignore=None, **kwargs):

        # gt_masks = [gt_mask.to_tensor(dtype=torch.bool, device=img.device) for gt_mask in gt_masks ]
        gt_masks = [torch.tensor(gt_mask,dtype=torch.bool, device=img.device) for gt_mask in gt_masks ]
        x = self.extract_feat(img)
        losses = dict()

        # CondInst and YOLACT have bbox_head
        if self.bbox_head:
            bbox_head_preds = self.bbox_head(x)         # bbox_head_preds is a tuple
            # positive_infos is a list of obj:`InstanceData`
            # It contains the information about the positive samples
            # CondInst, YOLACT
            det_losses, positive_infos = self.bbox_head.loss(   *bbox_head_preds,
                                                                gt_bboxes=gt_bboxes,
                                                                gt_labels=gt_labels,
                                                                gt_masks=gt_masks,
                                                                img_metas=img_metas,
                                                                gt_bboxes_ignore=gt_bboxes_ignore,
                                                                **kwargs)
            losses.update(det_losses)
        else:
            positive_infos = None

        mask_loss = self.mask_head( x[self.mask_head.start_level:self.mask_head.end_level+1],
                                    gt_labels, gt_masks, img_metas,
                                    positive_infos=positive_infos,
                                    gt_bboxes=gt_bboxes,
                                    gt_bboxes_ignore=gt_bboxes_ignore,
                                    **kwargs)
        # avoid loss override
        assert not set(mask_loss.keys()) & set(losses.keys())
        losses.update(mask_loss)

        return losses


    def forward_test(self, imgs, img_metas, rescale=False, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(name, type(var)))
        num_augs = len(imgs)

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)


    def simple_test(self, img, img_meta, rescale=False):
        feat = self.extract_feat(img)       # backbone + FPN
        if self.bbox_head:
            outs = self.bbox_head(feat)
            # results_list is list[obj:`InstanceData`]
            results_list = self.bbox_head.get_results(*outs, img_metas=img_meta, cfg=self.test_cfg, rescale=rescale)
        else:
            results_list = None

        results_list = self.mask_head.simple_test(feat, img_meta, rescale=rescale, instances_list=results_list)

        return results_list


    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation."""
        raise NotImplementedError





    #
    # def format_results(self, results):
    #     """Format the model predictions according to the interface with
    #     dataset.
    #
    #     Args:
    #         results (:obj:`InstanceData`): Processed
    #             results of single images. Usually contains
    #             following keys.
    #
    #             - scores (Tensor): Classification scores, has shape
    #               (num_instance,)
    #             - labels (Tensor): Has shape (num_instances,).
    #             - masks (Tensor): Processed mask results, has
    #               shape (num_instances, h, w).
    #
    #     Returns:
    #         tuple: Formatted bbox and mask results.. It contains two items:
    #
    #             - bbox_results (list[np.ndarray]): BBox results of
    #               single image. The list corresponds to each class.
    #               each ndarray has a shape (N, 5), N is the number of
    #               bboxes with this category, and last dimension
    #               5 arrange as (x1, y1, x2, y2, scores).
    #             - mask_results (list[np.ndarray]): Mask results of
    #               single image. The list corresponds to each class.
    #               each ndarray has shape (N, img_h, img_w), N
    #               is the number of masks with this category.
    #     """
    #     data_keys = results.keys()
    #     assert 'scores' in data_keys
    #     assert 'labels' in data_keys
    #
    #     assert 'masks' in data_keys, \
    #         'results should contain ' \
    #         'masks when format the results '
    #     mask_results = [[] for _ in range(self.mask_head.num_classes)]
    #
    #     num_masks = len(results)
    #
    #     if num_masks == 0:
    #         bbox_results = [np.zeros((0, 5), dtype=np.float32) for _ in range(self.mask_head.num_classes) ]
    #         return bbox_results, mask_results
    #
    #     labels = results.labels.detach().cpu().numpy()
    #
    #     if 'bboxes' not in results:
    #         # create dummy bbox results to store the scores
    #         results.bboxes = results.scores.new_zeros(len(results), 4)
    #
    #     det_bboxes = torch.cat([results.bboxes, results.scores[:, None]], dim=-1)
    #     det_bboxes = det_bboxes.detach().cpu().numpy()
    #     bbox_results = [det_bboxes[labels == i, :] for i in range(self.mask_head.num_classes)]
    #
    #     masks = results.masks.detach().cpu().numpy()
    #
    #     for idx in range(num_masks):
    #         mask = masks[idx]
    #         mask_results[labels[idx]].append(mask)
    #
    #     return bbox_results, mask_results