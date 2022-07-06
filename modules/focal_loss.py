

import torch
import torch.nn as nn
import torch.nn.functional as F
# from modules.sigmoid_focal_loss import sigmoid_focal_loss as _sigmoid_focal_loss
# from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
# from . import sigmoid_focal_loss2 as _sigmoid_focal_loss

# from torchvision.ops.focal_loss import sigmoid_focal_loss
from .utils import weight_reduce_loss



def py_sigmoid_focal_loss(pred, target, weight=None,gamma=3.0,alpha=0.75,
                          reduction='mean',num_clas=81 ,avg_factor=None):
    pred_sigmoid = pred.sigmoid()           # sigmoid二分类

    target = F.one_hot(target, num_classes=num_clas)
    target = target.type_as(pred)

    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss



def _sigmoid_focal_loss(ctx,
                        input,
                        target,
                        gamma=2.0,
                        alpha=0.25,
                        weight=None,
                        reduction='mean'):

    # assert isinstance(
    #     target, (torch.Tensor, torch.LongTensor, torch.cuda.LongTensor))
    assert input.dim() == 2
    assert target.dim() == 1
    assert input.size(0) == target.size(0)
    if weight is None:
        weight = input.new_empty(0)
    else:
        assert weight.dim() == 1
        assert input.size(1) == weight.size(0)
    ctx.reduction_dict = {'none': 0, 'mean': 1, 'sum': 2}
    assert reduction in ctx.reduction_dict.keys()

    ctx.gamma = float(gamma)
    ctx.alpha = float(alpha)
    ctx.reduction = ctx.reduction_dict[reduction]

    output = input.new_zeros(input.size())

    # ext_module.sigmoid_focal_loss_forward(
    #     input, target, weight, output, gamma=ctx.gamma, alpha=ctx.alpha)
    if ctx.reduction == ctx.reduction_dict['mean']:
        output = output.sum() / input.size(0)
    elif ctx.reduction == ctx.reduction_dict['sum']:
        output = output.sum()
    ctx.save_for_backward(input, target, weight)
    return output



class FocalLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma       = gamma
        self.alpha       = alpha
        self.reduction   = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, num_clas, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        '''
        print(pred.shape)
      
        print(target.shape)
        print(target)
        print(target.nonzero())
        idx = target.nonzero()
        print(target[idx])
        '''
#        print(weight,self.gamma,self.alpha,reduction,avg_factor)

        if self.use_sigmoid:

            loss_cls = self.loss_weight * py_sigmoid_focal_loss(
                pred, target, weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor,
                num_clas=num_clas)
        else:
            raise NotImplementedError
        return loss_cls
                                         
