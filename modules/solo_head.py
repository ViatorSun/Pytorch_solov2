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



import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

# from core import InstanceData, mask_matrix_nms, multi_apply
from .misc import multi_apply, matrix_nms, mask_matrix_nms
from .focal_loss import FocalLoss

# from core.utils import generate_coordinate
# from models.builder import HEADS
from .base_mask_head import BaseMaskHead
from .loss import FocalLoss, DiceLoss
from .nninit import xavier_init, kaiming_init, normal_init

from config import cfg
from mmdet.core.data_structures.instance_data import InstanceData




def generate_coordinate(featmap_sizes, device='cuda'):
    x_range = torch.linspace(-1, 1, featmap_sizes[-1], device=device)
    y_range = torch.linspace(-1, 1, featmap_sizes[-2], device=device)
    y, x = torch.meshgrid(y_range, x_range, indexing='ij')
    y = y.expand([featmap_sizes[0], 1, -1, -1])
    x = x.expand([featmap_sizes[0], 1, -1, -1])
    coord_feat = torch.cat([x, y], 1)
    return coord_feat


def center_of_mass(bitmasks):
    h, w = bitmasks.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ys = torch.arange(0, h, dtype=torch.float32, device=device)
    xs = torch.arange(0, w, dtype=torch.float32, device=device)

    m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
    m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
    m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
    center_x = m10 / m00
    center_y = m01 / m00
    return center_x, center_y



test_cfg = dict(
    nms_pre=500,
    score_thr=0.80,
    mask_thr=0.75,
    update_thr=0.05,
    kernel='gaussian',  # gaussian/linear
    sigma=2.0,
    max_per_img=30)



class SOLOHead(nn.Module):
    """SOLO mask head used in `SOLO: Segmenting Objects by Locations.

    <https://arxiv.org/abs/1912.04488>`_

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
            Default: 256.
        stacked_convs (int): Number of stacking convs of the head.
            Default: 4.
        strides (tuple): Downsample factor of each feature map.
        scale_ranges (tuple[tuple[int, int]]): Area range of multiple
            level masks, in the format [(min1, max1), (min2, max2), ...].
            A range of (16, 64) means the area range between (16, 64).
        pos_scale (float): Constant scale factor to control the center region.
        num_grids (list[int]): Divided image into a uniform grids, each
            feature map has a different grid value. The number of output
            channels is grid ** 2. Default: [40, 36, 24, 16, 12].
        cls_down_index (int): The index of downsample operation in
            classification branch. Default: 0.
        loss_mask (dict): Config of mask loss.
        loss_cls (dict): Config of classification loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32,
                                   requires_grad=True).
        train_cfg (dict): Training config of head.
        test_cfg (dict): Testing config of head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        start_level = 0,
        end_level   = 4,
        feat_channels   = 256,
        stacked_convs   = 4,
        strides         = (4, 8, 16, 32, 64),
        scale_ranges    = ((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
        pos_scale       = 0.2,
        num_grids       = [40, 36, 24, 16, 12],
        cls_down_index  = 0,
        norm_cfg        = dict(type='GN', num_groups=32, requires_grad=True),
        train_cfg       = None,
        test_cfg        = None,
    ):
        super(SOLOHead, self).__init__()
        self.num_classes        = num_classes
        self.cls_out_channels   = self.num_classes
        self.in_channels        = in_channels
        self.feat_channels      = feat_channels
        self.stacked_convs      = stacked_convs
        self.strides            = strides
        self.num_grids          = num_grids
        self.start_level        = start_level
        self.end_level          = end_level

        # number of FPN feats
        self.num_levels     = len(strides)
        assert self.num_levels == len(scale_ranges) == len(num_grids)
        self.scale_ranges   = scale_ranges
        self.pos_scale      = pos_scale

        self.cls_down_index = cls_down_index
        self.loss_cls       = FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0)
        self.loss_mask      = DiceLoss(use_sigmoid=True, loss_weight=3.0)
        self.norm_cfg       = norm_cfg
        self.train_cfg      = train_cfg
        self.test_cfg       = test_cfg

        self._init_layers()

    def _init_layers(self):
        self.mask_convs = nn.ModuleList()
        self.cate_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.feat_channels
            self.mask_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, norm_cfg=self.norm_cfg))
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cate_convs.append(ConvModule( chn, self.feat_channels, 3, stride=1, padding=1, norm_cfg=self.norm_cfg))

        self.conv_mask_list = nn.ModuleList()
        for num_grid in self.num_grids:
            self.conv_mask_list.append(nn.Conv2d(self.feat_channels, num_grid**2, 1))

        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)

    def init_weights(self):
        for m in self.mask_convs:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        normal_init(con, std=0.01)

        for m in self.cate_convs:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        normal_init(con, std=0.01)

        for m in self.conv_mask_list:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        normal_init(con, std=0.01)

        normal_init(self.conv_cls, std=0.01)


    def resize_feats(self, feats):
        """Downsample the first feat and upsample last feat in feats."""
        out = []
        for i in range(len(feats)):
            if i == 0:
                out.append(F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'))
            elif i == len(feats) - 1:
                out.append(F.interpolate( feats[i], size=feats[i - 1].shape[-2:], mode='bilinear'))
            else:
                out.append(feats[i])
        return out

    def forward(self, x,
                      gt_labels,
                      gt_masks,
                      img_metas,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      positive_infos=None,
                      **kwargs):

        if positive_infos is None:
            outs = self.forward_train(x)
        else:
            outs = self(x, positive_infos)

        assert isinstance(outs, tuple), 'Forward results should be a tuple, even if only one item is returned'
        loss = self.loss(*outs, gt_labels=gt_labels, gt_masks=gt_masks,
                         img_metas=img_metas, gt_bboxes=gt_bboxes,
                         gt_bboxes_ignore=gt_bboxes_ignore,
                         positive_infos=positive_infos, **kwargs)
        return loss

    def forward_train(self, feats):
        assert len(feats) == self.num_levels
        feats = self.resize_feats(feats)
        mlvl_mask_preds = []
        mlvl_cate_preds = []
        for i in range(self.num_levels):
            x = feats[i]
            mask_feat = x
            cls_feat  = x

            # generate and concat the coordinate
            coord_feat = generate_coordinate(mask_feat.size(),mask_feat.device)
            mask_feat  = torch.cat([mask_feat, coord_feat], 1)

            for mask_layer in self.mask_convs:
                mask_feat = mask_layer(mask_feat)

            mask_feat = F.interpolate(mask_feat, scale_factor=2, mode='bilinear')
            mask_pred = self.conv_mask_list[i](mask_feat)

            # cate cls branch
            for j, cls_layer in enumerate(self.cate_convs):
                if j == self.cls_down_index:
                    num_grid = self.num_grids[i]
                    cls_feat = F.interpolate(cls_feat, size=num_grid, mode='bilinear')
                cls_feat = cls_layer(cls_feat)

            cls_pred = self.conv_cls(cls_feat)

            if not self.training:
                feat_wh = feats[0].size()[-2:]
                upsampled_size = (feat_wh[0] * 2, feat_wh[1] * 2)
                mask_pred = F.interpolate(mask_pred.sigmoid(), size=upsampled_size, mode='bilinear')
                cls_pred = cls_pred.sigmoid()
                # get local maximum
                local_max = F.max_pool2d(cls_pred, 2, stride=1, padding=1)
                keep_mask = local_max[:, :, :-1, :-1] == cls_pred
                cls_pred = cls_pred * keep_mask

            mlvl_mask_preds.append(mask_pred)
            mlvl_cate_preds.append(cls_pred)
        return mlvl_mask_preds, mlvl_cate_preds


    def loss(self, mlvl_mask_preds, mlvl_cls_preds, gt_labels, gt_masks,
             img_metas, gt_bboxes=None, **kwargs):
        """Calculate the loss of total batch.

        Args:
            mlvl_mask_preds (list[Tensor]): Multi-level mask prediction.
                Each element in the list has shape
                (batch_size, num_grids**2 ,h ,w).
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
        num_imgs = len(gt_labels)

        featmap_sizes = [featmap.size()[-2:] for featmap in mlvl_mask_preds]

        # `BoolTensor` in `pos_masks` represent
        # whether the corresponding point is
        # positive
        pos_mask_targets, labels, pos_masks = multi_apply(
            self._get_targets_single,
            gt_bboxes,
            gt_labels,
            gt_masks,
            featmap_sizes=featmap_sizes)

        # change from the outside list meaning multi images
        # to the outside list meaning multi levels
        mlvl_pos_mask_targets = [[] for _ in range(num_levels)]
        mlvl_pos_mask_preds = [[] for _ in range(num_levels)]
        mlvl_pos_masks = [[] for _ in range(num_levels)]
        mlvl_labels = [[] for _ in range(num_levels)]

        for img_id in range(num_imgs):
            assert num_levels == len(pos_mask_targets[img_id])
            for lvl in range(num_levels):
                mlvl_pos_mask_targets[lvl].append(pos_mask_targets[img_id][lvl])
                mlvl_pos_mask_preds[lvl].append(mlvl_mask_preds[lvl][img_id, pos_masks[img_id][lvl], ...])
                mlvl_pos_masks[lvl].append(pos_masks[img_id][lvl].flatten())
                mlvl_labels[lvl].append(labels[img_id][lvl].flatten())

        # cat multiple image
        temp_mlvl_cls_preds = []
        for lvl in range(num_levels):
            mlvl_pos_mask_targets[lvl] = torch.cat(mlvl_pos_mask_targets[lvl], dim=0)
            mlvl_pos_mask_preds[lvl]   = torch.cat(mlvl_pos_mask_preds[lvl], dim=0)
            mlvl_pos_masks[lvl]        = torch.cat(mlvl_pos_masks[lvl], dim=0)
            mlvl_labels[lvl]           = torch.cat(mlvl_labels[lvl], dim=0)
            temp_mlvl_cls_preds.append(mlvl_cls_preds[lvl].permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels))

        num_pos = sum(item.sum() for item in mlvl_pos_masks)
        # dice loss
        loss_mask = []
        for pred, target in zip(mlvl_pos_mask_preds, mlvl_pos_mask_targets):
            if pred.size()[0] == 0:
                loss_mask.append(pred.sum().unsqueeze(0))
                continue
            loss_mask.append(self.loss_mask(pred, target, reduction_override='none'))
        if num_pos > 0:
            loss_mask = torch.cat(loss_mask).sum() / num_pos
        else:
            loss_mask = torch.cat(loss_mask).mean()

        flatten_labels = torch.cat(mlvl_labels)
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
                - mlvl_pos_masks (list[Tensor]): Each element is
                  a `BoolTensor` to represent whether the
                  corresponding point in single level
                  is positive, has shape (num_grid **2).
        """
        device = gt_labels.device
        gt_areas = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1]))

        mlvl_pos_mask_targets = []
        mlvl_labels = []
        mlvl_pos_masks = []
        for (lower_bound, upper_bound), stride, featmap_size, num_grid \
                in zip(self.scale_ranges, self.strides, featmap_sizes, self.num_grids):

            mask_target = torch.zeros([num_grid**2, featmap_size[0], featmap_size[1]], dtype=torch.uint8, device=device)
            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
            labels = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device) + self.num_classes
            pos_mask = torch.zeros([num_grid**2], dtype=torch.bool, device=device)

            gt_inds = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            if len(gt_inds) == 0:
                mlvl_pos_mask_targets.append(mask_target.new_zeros(0, featmap_size[0], featmap_size[1]))
                mlvl_labels.append(labels)
                mlvl_pos_masks.append(pos_mask)
                continue
            hit_gt_bboxes = gt_bboxes[gt_inds]
            hit_gt_labels = gt_labels[gt_inds]
            hit_gt_masks  = gt_masks[gt_inds, ...]

            pos_w_ranges = 0.5 * (hit_gt_bboxes[:, 2] - hit_gt_bboxes[:, 0]) * self.pos_scale
            pos_h_ranges = 0.5 * (hit_gt_bboxes[:, 3] - hit_gt_bboxes[:, 1]) * self.pos_scale

            # Make sure hit_gt_masks has a value
            valid_mask_flags = hit_gt_masks.sum(dim=-1).sum(dim=-1) > 0
            output_stride    = stride / 2

            for gt_mask, gt_label, half_h, half_w, valid_mask_flag in \
                    zip(hit_gt_masks, hit_gt_labels, pos_h_ranges, pos_w_ranges, valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (featmap_sizes[0][0] * 4, featmap_sizes[0][1] * 4)
                center_h, center_w = center_of_mass(gt_mask)


                coord_w = int(torch.div((center_w / upsampled_size[1]), (1. / num_grid), rounding_mode='floor'))
                coord_h = int(torch.div((center_h / upsampled_size[0]), (1. / num_grid), rounding_mode='floor'))

                # left, top, right, down
                _ = num_grid - 1
                top_box   = max(0, int(torch.div(((center_h - half_h) / upsampled_size[0]), (1. / num_grid), rounding_mode='floor')))
                down_box  = min(_, int(torch.div(((center_h + half_h) / upsampled_size[0]), (1. / num_grid), rounding_mode='floor')))
                left_box  = max(0, int(torch.div(((center_w - half_w) / upsampled_size[1]), (1. / num_grid), rounding_mode='floor')))
                right_box = min(_, int(torch.div(((center_w + half_w) / upsampled_size[1]), (1. / num_grid), rounding_mode='floor')))

                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)

                labels[top:(down + 1), left:(right + 1)] = gt_label
                # ins
                gt_mask = np.uint8(gt_mask.cpu().numpy())
                # Follow the original implementation, F.interpolate is
                # different from cv2 and opencv
                gt_mask = mmcv.imrescale(gt_mask, scale=1. / output_stride)
                gt_mask = torch.from_numpy(gt_mask).to(device=device)

                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        index = int(i * num_grid + j)
                        mask_target[index, :gt_mask.shape[0], :gt_mask.shape[1]] = gt_mask
                        pos_mask[index] = True
            mlvl_pos_mask_targets.append(mask_target[pos_mask])
            mlvl_labels.append(labels)
            mlvl_pos_masks.append(pos_mask)
        return mlvl_pos_mask_targets, mlvl_labels, mlvl_pos_masks


    """ test modules """
    def simple_test(self, feats, img_metas, rescale=False, instances_list=None, **kwargs):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
            instances_list (list[obj:`InstanceData`], optional): Detection
                results of each image after the post process. Only exist
                if there is a `bbox_head`, like `YOLACT`, `CondInst`, etc.

        Returns:
            list[obj:`InstanceData`]: Instance segmentation \
                results of each image after the post process. \
                Each item usually contains following keys. \

                - scores (Tensor): Classification scores, has a shape
                  (num_instance,)
                - labels (Tensor): Has a shape (num_instances,).
                - masks (Tensor): Processed mask results, has a
                  shape (num_instances, h, w).
        """
        # if instances_list is None:
        #     outs = self.forward_train(feats)
        # else:
        outs = self.forward_train(feats)
        mask_pred = outs + (img_metas,)
        results_list = self.get_results(*mask_pred,
                                        rescale=rescale,
                                        instances_list=instances_list,
                                        **kwargs)
        return results_list


    def get_results(self, mask_preds, cls_scores, img_metas, **kwargs):
        """Get multi-image mask results.

        Args:
            mask_preds (list[Tensor]): Multi-level mask prediction.
                Each element in the list has shape
                (batch_size, num_grids**2 ,h ,w).
            cls_scores (list[Tensor]): Multi-level scores. Each element
                in the list has shape
                (batch_size, num_classes, num_grids ,num_grids).
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
        cls_scores = [ item.permute(0, 2, 3, 1) for item in cls_scores ]    #  -> [batch, grid, grid, class]
        assert len(mask_preds) == len(cls_scores)
        num_levels = len(cls_scores)

        results_list = []
        for img_id in range(len(img_metas)):
            cls_pred_list  = [cls_scores[lvl][img_id].view(-1, self.cls_out_channels) for lvl in range(num_levels) ]
            mask_pred_list = [mask_preds[lvl][img_id] for lvl in range(num_levels) ]

            cls_pred_list  = torch.cat(cls_pred_list, dim=0)
            mask_pred_list = torch.cat(mask_pred_list, dim=0)

            results = self.get_results_single(cls_pred_list, mask_pred_list, img_meta=img_metas[img_id])
            results_list.append(results)

        return results_list

    def get_results_single(self, cls_scores, mask_preds, img_meta, cfg=None):
        """Get processed mask related results of single image.

        Args:
            cls_scores (Tensor): Classification score of all points
                in single image, has shape (num_points, num_classes).
            mask_preds (Tensor): Mask prediction of all points in
                single image, has shape (num_points, feat_h, feat_w).
            img_meta (dict): Meta information of corresponding image.
            cfg (dict, optional): Config used in test phase.
                Default: None.

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


        # cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(mask_preds)
        results = InstanceData(img_meta)

        featmap_size = mask_preds.size()[-2:]

        img_shape = results.img_shape
        ori_shape = results.ori_shape

        h, w, _ = img_shape
        upsampled_size = (featmap_size[0] * 4, featmap_size[1] * 4)

        score_mask = (cls_scores > 0.2 ) # cfg.score_thr)
        cls_scores = cls_scores[score_mask]
        if len(cls_scores) == 0:
            return empty_results(results, cls_scores)

        inds = score_mask.nonzero()
        cls_labels = inds[:, 1]

        # Filter the mask with an area is smaller than
        # stride of corresponding feature level
        interval = cls_labels.new_tensor(self.num_grids).pow(2).cumsum(0)       # 间隔/区间
        strides  = cls_scores.new_ones(interval[-1])
        strides[:interval[0]] *= self.strides[0]
        for level in range(1, self.num_levels):
            strides[interval[level - 1]:interval[level]] *= self.strides[level]
        strides = strides[inds[:, 0]]

        # mask
        mask_preds = mask_preds[inds[:, 0]]

        masks = mask_preds > 0.75   # cfg.mask_thr
        sum_masks = masks.sum((1, 2)).float()
        keep = sum_masks > strides
        if keep.sum() == 0:
            return empty_results(results, cls_scores)
        masks = masks[keep]
        mask_preds = mask_preds[keep]
        sum_masks  = sum_masks[keep]
        cls_scores = cls_scores[keep]
        cls_labels = cls_labels[keep]

        # maskness.
        mask_scores = (mask_preds * masks).sum((1, 2)) / sum_masks
        cls_scores *= mask_scores

        scores, labels, _, keep_inds = mask_matrix_nms( masks, cls_labels, cls_scores,
                                                        mask_area  = sum_masks,
                                                        nms_pre    = 500 ,  # cfg.nms_pre,
                                                        max_num    = 30  ,  # cfg.max_per_img,
                                                        kernel     = 'gaussian', #cfg.kernel,
                                                        sigma      = 2.0 , # cfg.sigma,
                                                        filter_thr = 0.05) #cfg.filter_thr)

        mask_preds     = mask_preds[keep_inds]
        mask_preds     = F.interpolate( mask_preds.unsqueeze(0), size=upsampled_size, mode='bilinear')[:, :, :h, :w]
        mask_preds     = F.interpolate( mask_preds, size=ori_shape[:2], mode='bilinear').squeeze(0)
        masks          = mask_preds > 0.75 # cfg.mask_thr

        # results.masks  = masks
        # results.labels = labels
        # results.scores = scores

        # return results
        return masks, labels, scores



