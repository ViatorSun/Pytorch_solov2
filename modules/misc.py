from functools import partial
from six.moves import map, zip
import torch 




def matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS for multi-class masks.

    Args:
        seg_masks (Tensor): shape (n, h, w)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss' 
        sigma (float): std in gaussian method
        sum_masks (Tensor): The sum of seg_masks

    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).float()
    seg_masks = seg_masks.reshape(n_samples, -1).float()

    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))

    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)

    # iou.
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)

    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay 
    decay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update






def mask_matrix_nms(masks, labels, scores, filter_thr=-1, nms_pre=-1,
                    max_num=-1, kernel='gaussian', sigma=2.0, mask_area=None):
    """Matrix NMS for multi-class masks.

    Args:
        masks (Tensor): Has shape (num_instances, h, w)
        labels (Tensor): Labels of corresponding masks,
            has shape (num_instances,).
        scores (Tensor): Mask scores of corresponding masks,
            has shape (num_instances).
        filter_thr (float): Score threshold to filter the masks
            after matrix nms. Default: -1, which means do not
            use filter_thr.
        nms_pre (int): The max number of instances to do the matrix nms.
            Default: -1, which means do not use nms_pre.
        max_num (int, optional): If there are more than max_num masks after
            matrix, only top max_num will be kept. Default: -1, which means
            do not use max_num.
        kernel (str): 'linear' or 'gaussian'.
        sigma (float): std in gaussian method.
        mask_area (Tensor): The sum of seg_masks.

    Returns:
        tuple(Tensor): Processed mask results.

            - scores (Tensor): Updated scores, has shape (n,).
            - labels (Tensor): Remained labels, has shape (n,).
            - masks (Tensor): Remained masks, has shape (n, w, h).
            - keep_inds (Tensor): The indices number of
                the remaining mask in the input mask, has shape (n,).
    """
    assert len(labels) == len(masks) == len(scores)
    if len(labels) == 0:
        return scores.new_zeros(0), labels.new_zeros(0), masks.new_zeros(
            0, *masks.shape[-2:]), labels.new_zeros(0)
    if mask_area is None:
        mask_area = masks.sum((1, 2)).float()
    else:
        assert len(masks) == len(mask_area)

    # sort and keep top nms_pre
    scores, sort_inds = torch.sort(scores, descending=True)

    keep_inds = sort_inds
    if (nms_pre > 0) and len(sort_inds) > nms_pre:
        sort_inds = sort_inds[:nms_pre]
        keep_inds = keep_inds[:nms_pre]
        scores = scores[:nms_pre]
    masks = masks[sort_inds]
    mask_area = mask_area[sort_inds]
    labels = labels[sort_inds]

    num_masks = len(labels)
    flatten_masks = masks.reshape(num_masks, -1).float()
    # inter.
    inter_matrix = torch.mm(flatten_masks, flatten_masks.transpose(1, 0))
    expanded_mask_area = mask_area.expand(num_masks, num_masks)
    # Upper triangle iou matrix.
    iou_matrix = (inter_matrix /
                  (expanded_mask_area + expanded_mask_area.transpose(1, 0) -
                   inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    expanded_labels = labels.expand(num_masks, num_masks)
    # Upper triangle label matrix.
    label_matrix = (expanded_labels == expanded_labels.transpose(1, 0)).triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(num_masks, num_masks).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # Calculate the decay_coefficient
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError(
            f'{kernel} kernel is not supported in matrix nms!')
    # update the score.
    scores = scores * decay_coefficient

    if filter_thr > 0:
        keep = scores >= filter_thr
        keep_inds = keep_inds[keep]
        if not keep.any():
            return scores.new_zeros(0), labels.new_zeros(0), masks.new_zeros(
                0, *masks.shape[-2:]), labels.new_zeros(0)
        masks = masks[keep]
        scores = scores[keep]
        labels = labels[keep]

    # sort and keep top max_num
    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = keep_inds[sort_inds]
    if (max_num > 0) and (len(sort_inds) > max_num):
        sort_inds = sort_inds[:max_num]
        keep_inds = keep_inds[:max_num]
        scores = scores[:max_num]
    masks = masks[sort_inds]
    labels = labels[sort_inds]

    return scores, labels, masks, keep_inds






def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
            map the multiple outputs of the ``func`` into different list.
            Each list contains the same type of outputs corresponding to different inputs.

    Args:
        func (Function): A function that will be applied to a list of arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))
