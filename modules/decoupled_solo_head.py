#  !/usr/bin/env  python
#  -*- coding:utf-8 -*-
# @Time     :  2022.08
# @Author   :  绿色羽毛
# @Email    :  lvseyumao@foxmail.com
# @Blog     :  https://blog.csdn.net/ViatorSun
# @arXiv    :   
# @version  :   
# @Note     :   
#
#


# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


from .misc import multi_apply, matrix_nms, mask_matrix_nms

from .base_mask_head import BaseMaskHead
from .loss import FocalLoss, DiceLoss
from .nninit import xavier_init, kaiming_init, normal_init


from mmdet.core.data_structures.instance_data import InstanceData
from .solo_head import SOLOHead
from .utils import generate_coordinate






class DecoupledSOLOHead(SOLOHead):
    """Decoupled SOLO mask head used in `SOLO: Segmenting Objects by Locations.

    <https://arxiv.org/abs/1912.04488>`_

    Args:
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 *args,
                 init_cfg=[
                     dict(type='Normal', layer='Conv2d', std=0.01),
                     dict(
                         type='Normal',
                         std=0.01,
                         bias_prob=0.01,
                         override=dict(name='conv_mask_list_x')),
                     dict(
                         type='Normal',
                         std=0.01,
                         bias_prob=0.01,
                         override=dict(name='conv_mask_list_y')),
                     dict(
                         type='Normal',
                         std=0.01,
                         bias_prob=0.01,
                         override=dict(name='conv_cls'))
                 ],
                 **kwargs):
        super(DecoupledSOLOHead, self).__init__(*args, **kwargs)

    def _init_layers(self):
        self.cls_convs    = nn.ModuleList()
        self.mask_convs_x = nn.ModuleList()
        self.mask_convs_y = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels + 1 if i == 0 else self.feat_channels
            self.mask_convs_x.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, norm_cfg=self.norm_cfg))
            self.mask_convs_y.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, norm_cfg=self.norm_cfg))

            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, norm_cfg=self.norm_cfg))

        self.conv_mask_list_x = nn.ModuleList()
        self.conv_mask_list_y = nn.ModuleList()
        for num_grid in self.num_grids:
            self.conv_mask_list_x.append(nn.Conv2d(self.feat_channels, num_grid, 3, padding=1))
            self.conv_mask_list_y.append(nn.Conv2d(self.feat_channels, num_grid, 3, padding=1))
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)

    def init_weights(self):
        for m in self.mask_convs_x:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        normal_init(con, std=0.01)

        for m in self.mask_convs_y:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        normal_init(con, std=0.01)

        for m in self.cls_convs:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        normal_init(con, std=0.01)

        for m in self.conv_mask_list_x:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        normal_init(con, std=0.01)

        for m in self.conv_mask_list_y:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        normal_init(con, std=0.01)

        normal_init(self.conv_cls, std=0.01)


    def forward(self, x,
                      gt_labels,
                      gt_masks,
                      img_metas,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      positive_infos=None,
                      **kwargs):

        # if positive_infos is None:
        #     outs = self.forward_train(x)
        # else:
        #     outs = self(x, positive_infos)

        outs = self.forward_train(x)

        assert isinstance(outs, tuple), 'Forward results should be a tuple, even if only one item is returned'

        loss = self.loss(*outs, gt_labels=gt_labels, gt_masks=gt_masks,
                         img_metas=img_metas, gt_bboxes=gt_bboxes,
                         gt_bboxes_ignore=gt_bboxes_ignore,
                         positive_infos=positive_infos, **kwargs)
        return loss

    def forward_train(self, feats):
        assert len(feats) == self.num_levels
        feats = self.resize_feats(feats)
        mask_preds_x = []
        mask_preds_y = []
        cls_preds = []
        for i in range(self.num_levels):
            x = feats[i]
            mask_feat = x
            cls_feat  = x

            # generate and concat the coordinate
            coord_feat  = generate_coordinate(mask_feat.size(), mask_feat.device)
            mask_feat_x = torch.cat([mask_feat, coord_feat[:, 0:1, ...]], 1)
            mask_feat_y = torch.cat([mask_feat, coord_feat[:, 1:2, ...]], 1)

            for mask_layer_x, mask_layer_y in zip(self.mask_convs_x, self.mask_convs_y):
                mask_feat_x = mask_layer_x(mask_feat_x)
                mask_feat_y = mask_layer_y(mask_feat_y)

            mask_feat_x = F.interpolate(mask_feat_x, scale_factor=2, mode='bilinear')
            mask_feat_y = F.interpolate(mask_feat_y, scale_factor=2, mode='bilinear')

            mask_pred_x = self.conv_mask_list_x[i](mask_feat_x)
            mask_pred_y = self.conv_mask_list_y[i](mask_feat_y)

            # cls branch
            for j, cls_layer in enumerate(self.cls_convs):
                if j == self.cls_down_index:
                    num_grid = self.num_grids[i]
                    cls_feat = F.interpolate(cls_feat, size=num_grid, mode='bilinear')
                cls_feat = cls_layer(cls_feat)

            cls_pred = self.conv_cls(cls_feat)

            if not self.training:
                feat_wh         = feats[0].size()[-2:]
                upsampled_size  = (feat_wh[0] * 2, feat_wh[1] * 2)
                mask_pred_x     = F.interpolate(mask_pred_x.sigmoid(), size=upsampled_size, mode='bilinear')
                mask_pred_y     = F.interpolate(mask_pred_y.sigmoid(), size=upsampled_size, mode='bilinear')
                cls_pred        = cls_pred.sigmoid()
                # get local maximum
                local_max   = F.max_pool2d(cls_pred, 2, stride=1, padding=1)
                keep_mask   = local_max[:, :, :-1, :-1] == cls_pred
                cls_pred    = cls_pred * keep_mask

            mask_preds_x.append(mask_pred_x)
            mask_preds_y.append(mask_pred_y)
            cls_preds.append(cls_pred)
        return mask_preds_x, mask_preds_y, cls_preds

    def loss(self,
             mlvl_mask_preds_x,
             mlvl_mask_preds_y,
             mlvl_cls_preds,
             gt_labels,
             gt_masks,
             img_metas,
             gt_bboxes=None,
             **kwargs):
        """Calculate the loss of total batch.

        Args:
            mlvl_mask_preds_x (list[Tensor]): Multi-level mask prediction
                from x branch. Each element in the list has shape
                (batch_size, num_grids ,h ,w).
            mlvl_mask_preds_y (list[Tensor]): Multi-level mask prediction
                from y branch. Each element in the list has shape
                (batch_size, num_grids ,h ,w).
            mlvl_cls_preds (list[Tensor]): Multi-level scores. Each element
                in the list has shape
                (batch_size, num_classes, num_grids ,num_grids).
            gt_labels (list[Tensor]): Labels of multiple images.
            gt_masks (list[Tensor]): Ground truth masks of multiple images.
                Each has shape (num_instances, h, w).
            img_metas (list[dict]): Meta information of multiple images.
            gt_bboxes (list[Tensor]): Ground truth bboxes of multiple
                images. Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        num_levels = self.num_levels
        num_imgs   = len(gt_labels)
        featmap_sizes = [featmap.size()[-2:] for featmap in mlvl_mask_preds_x]

        pos_mask_targets, labels, xy_pos_indexes =multi_apply(self._get_targets_single,
                                                              gt_bboxes, gt_labels, gt_masks,
                                                              featmap_sizes=featmap_sizes)

        # change from the outside list meaning multi images to the outside list meaning multi levels
        mlvl_pos_mask_targets = [[] for _ in range(num_levels)]
        mlvl_pos_mask_preds_x = [[] for _ in range(num_levels)]
        mlvl_pos_mask_preds_y = [[] for _ in range(num_levels)]
        mlvl_labels           = [[] for _ in range(num_levels)]

        for img_id in range(num_imgs):
            for lvl in range(num_levels):
                mlvl_pos_mask_targets[lvl].append(pos_mask_targets[img_id][lvl])
                mlvl_pos_mask_preds_x[lvl].append(mlvl_mask_preds_x[lvl][img_id, xy_pos_indexes[img_id][lvl][:, 1]])
                mlvl_pos_mask_preds_y[lvl].append(mlvl_mask_preds_y[lvl][img_id, xy_pos_indexes[img_id][lvl][:, 0]])
                mlvl_labels[lvl].append(labels[img_id][lvl].flatten())

        # cat multiple image
        temp_mlvl_cls_preds = []
        for lvl in range(num_levels):
            mlvl_pos_mask_targets[lvl] = torch.cat(mlvl_pos_mask_targets[lvl], dim=0)
            mlvl_pos_mask_preds_x[lvl] = torch.cat(mlvl_pos_mask_preds_x[lvl], dim=0)
            mlvl_pos_mask_preds_y[lvl] = torch.cat(mlvl_pos_mask_preds_y[lvl], dim=0)
            mlvl_labels[lvl] = torch.cat(mlvl_labels[lvl], dim=0)
            temp_mlvl_cls_preds.append(mlvl_cls_preds[lvl].permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels))

        # dice loss
        num_pos = 0.
        loss_mask = []
        for pred_x, pred_y, target in zip(mlvl_pos_mask_preds_x, mlvl_pos_mask_preds_y, mlvl_pos_mask_targets):
            num_masks = pred_x.size(0)
            if num_masks == 0:
                # make sure can get grad
                loss_mask.append((pred_x.sum() + pred_y.sum()).unsqueeze(0))
                continue
            num_pos  += num_masks
            pred_mask = pred_x.sigmoid() * pred_y.sigmoid()
            assert (pred_mask.shape == target.shape), 'pred_mask.shape != target.shape'
            loss_mask.append(self.loss_mask(pred_mask, target, reduction_override='none'))

        if num_pos > 0:
            loss_mask = torch.cat(loss_mask).sum() / num_pos
        else:
            loss_mask = torch.cat(loss_mask).mean()

        # cate
        flatten_labels    = torch.cat(mlvl_labels)
        flatten_cls_preds = torch.cat(temp_mlvl_cls_preds)

        loss_cls = self.loss_cls(flatten_cls_preds, flatten_labels, avg_factor=num_pos + 1)
        return dict(loss_mask=loss_mask, loss_cate=loss_cls)

    def _get_targets_single(self, gt_bboxes, gt_labels, gt_masks, featmap_sizes=None):
        """Compute targets for predictions of single image.

        Args:
            gt_bboxes (Tensor): Ground truth bbox of each instance,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth label of each instance,
                shape (num_gts,).
            gt_masks (Tensor): Ground truth mask of each instance,
                shape (num_gts, h, w).
            featmap_sizes (list[:obj:`torch.size`]): Size of each
                feature map from feature pyramid, each element
                means (feat_h, feat_w). Default: None.

        Returns:
            Tuple: Usually returns a tuple containing targets for predictions.

                - mlvl_pos_mask_targets (list[Tensor]): Each element represent
                  the binary mask targets for positive points in this
                  level, has shape (num_pos, out_h, out_w).
                - mlvl_labels (list[Tensor]): Each element is
                  classification labels for all
                  points in this level, has shape
                  (num_grid, num_grid).
                - mlvl_xy_pos_indexes (list[Tensor]): Each element
                  in the list contains the index of positive samples in
                  corresponding level, has shape (num_pos, 2), last
                  dimension 2 present (index_x, index_y).
        """
        mlvl_pos_mask_targets, mlvl_labels, mlvl_pos_masks = super()._get_targets_single(
                                                            gt_bboxes, gt_labels, gt_masks,
                                                            featmap_sizes=featmap_sizes)

        mlvl_xy_pos_indexes = [(self.num_classes - item).nonzero() for item in mlvl_labels]

        return mlvl_pos_mask_targets, mlvl_labels, mlvl_xy_pos_indexes

    def get_results(self,
                    mlvl_mask_preds_x,
                    mlvl_mask_preds_y,
                    mlvl_cls_scores,
                    img_metas,
                    rescale=None,
                    **kwargs):
        """Get multi-image mask results.

        Args:
            mlvl_mask_preds_x (list[Tensor]): Multi-level mask prediction
                from x branch. Each element in the list has shape
                (batch_size, num_grids ,h ,w).
            mlvl_mask_preds_y (list[Tensor]): Multi-level mask prediction
                from y branch. Each element in the list has shape
                (batch_size, num_grids ,h ,w).
            mlvl_cls_scores (list[Tensor]): Multi-level scores. Each element
                in the list has shape
                (batch_size, num_classes ,num_grids ,num_grids).
            img_metas (list[dict]): Meta information of all images.

        Returns:
            list[:obj:`InstanceData`]: Processed results of multiple
            images.Each :obj:`InstanceData` usually contains
            following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,).
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).
        """
        mlvl_cls_scores = [item.permute(0, 2, 3, 1) for item in mlvl_cls_scores]
        assert len(mlvl_mask_preds_x) == len(mlvl_cls_scores)
        num_levels = len(mlvl_cls_scores)

        results_list = []
        for img_id in range(len(img_metas)):
            cls_pred_list    = [mlvl_cls_scores[i][img_id].view(-1, self.cls_out_channels).detach() for i in range(num_levels) ]
            mask_pred_list_x = [mlvl_mask_preds_x[i][img_id] for i in range(num_levels) ]
            mask_pred_list_y = [mlvl_mask_preds_y[i][img_id] for i in range(num_levels) ]

            cls_pred_list    = torch.cat(cls_pred_list, dim=0)
            mask_pred_list_x = torch.cat(mask_pred_list_x, dim=0)
            mask_pred_list_y = torch.cat(mask_pred_list_y, dim=0)

            results = self.get_results_single(
                cls_pred_list,
                mask_pred_list_x,
                mask_pred_list_y,
                img_meta=img_metas[img_id],
                cfg=self.test_cfg)
            results_list.append(results)
        return results_list

    def get_results_single(self, cls_scores, mask_preds_x, mask_preds_y, img_meta, cfg):
        """Get processed mask related results of single image.

        Args:
            cls_scores (Tensor): Classification score of all points
                in single image, has shape (num_points, num_classes).
            mask_preds_x (Tensor): Mask prediction of x branch of
                all points in single image, has shape
                (sum_num_grids, feat_h, feat_w).
            mask_preds_y (Tensor): Mask prediction of y branch of
                all points in single image, has shape
                (sum_num_grids, feat_h, feat_w).
            img_meta (dict): Meta information of corresponding image.
            cfg (dict): Config used in test phase.

        Returns:
            :obj:`InstanceData`: Processed results of single image.
             it usually contains following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,).
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).
        """

        def empty_results(results, cls_scores):
            """Generate a empty results."""
            results.scores = cls_scores.new_ones(0)
            results.masks  = cls_scores.new_zeros(0, *results.ori_shape[:2])
            results.labels = cls_scores.new_ones(0)
            return results

        cfg = self.test_cfg if cfg is None else cfg

        results = InstanceData(img_meta)
        img_shape = results.img_shape
        ori_shape = results.ori_shape
        h, w, _ = img_shape
        featmap_size = mask_preds_x.size()[-2:]
        upsampled_size = (featmap_size[0] * 4, featmap_size[1] * 4)

        score_mask = (cls_scores > cfg.score_thr)
        cls_scores = cls_scores[score_mask]
        inds = score_mask.nonzero()
        lvl_interval = inds.new_tensor(self.num_grids).pow(2).cumsum(0)
        num_all_points = lvl_interval[-1]
        lvl_start_index = inds.new_ones(num_all_points)
        num_grids = inds.new_ones(num_all_points)
        seg_size = inds.new_tensor(self.num_grids).cumsum(0)
        mask_lvl_start_index = inds.new_ones(num_all_points)
        strides = inds.new_ones(num_all_points)

        lvl_start_index[:lvl_interval[0]] *= 0
        mask_lvl_start_index[:lvl_interval[0]] *= 0
        num_grids[:lvl_interval[0]] *= self.num_grids[0]
        strides[:lvl_interval[0]] *= self.strides[0]

        for lvl in range(1, self.num_levels):
            lvl_start_index[lvl_interval[lvl - 1]:lvl_interval[lvl]] *= lvl_interval[lvl - 1]
            mask_lvl_start_index[lvl_interval[lvl - 1]:lvl_interval[lvl]] *= seg_size[lvl - 1]
            num_grids[lvl_interval[lvl - 1]:lvl_interval[lvl]] *= self.num_grids[lvl]
            strides[lvl_interval[lvl - 1]:lvl_interval[lvl]] *= self.strides[lvl]

        lvl_start_index = lvl_start_index[inds[:, 0]]
        mask_lvl_start_index = mask_lvl_start_index[inds[:, 0]]
        num_grids = num_grids[inds[:, 0]]
        strides = strides[inds[:, 0]]

        y_lvl_offset = (inds[:, 0] - lvl_start_index) // num_grids
        x_lvl_offset = (inds[:, 0] - lvl_start_index) % num_grids
        y_inds = mask_lvl_start_index + y_lvl_offset
        x_inds = mask_lvl_start_index + x_lvl_offset

        cls_labels = inds[:, 1]
        mask_preds = mask_preds_x[x_inds, ...] * mask_preds_y[y_inds, ...]

        masks = mask_preds > cfg.mask_thr
        sum_masks = masks.sum((1, 2)).float()
        keep = sum_masks > strides
        if keep.sum() == 0:
            return empty_results(results, cls_scores)

        masks = masks[keep]
        mask_preds = mask_preds[keep]
        sum_masks = sum_masks[keep]
        cls_scores = cls_scores[keep]
        cls_labels = cls_labels[keep]

        # maskness.
        mask_scores = (mask_preds * masks).sum((1, 2)) / sum_masks
        cls_scores *= mask_scores

        scores, labels, _, keep_inds = mask_matrix_nms(
            masks,
            cls_labels,
            cls_scores,
            mask_area=sum_masks,
            nms_pre=cfg.nms_pre,
            max_num=cfg.max_per_img,
            kernel=cfg.kernel,
            sigma=cfg.sigma,
            filter_thr=cfg.filter_thr)
        mask_preds = mask_preds[keep_inds]
        mask_preds = F.interpolate(mask_preds.unsqueeze(0), size=upsampled_size, mode='bilinear')[:, :, :h, :w]
        mask_preds = F.interpolate(mask_preds, size=ori_shape[:2], mode='bilinear').squeeze(0)
        masks = mask_preds > cfg.mask_thr

        results.masks  = masks
        results.labels = labels
        results.scores = scores

        return results