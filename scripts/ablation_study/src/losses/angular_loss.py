import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class AngularLoss(_Loss):
    '''
    angular loss, stable between 0 and 360 degree
    '''

    def __init__(self,
                 spatial_dimensions=3,
                 is_volume_weighted = False, 
                 eps=1e-6,
                 reduction='mean'):
        super(AngularLoss, self).__init__(reduction=reduction)
        self.eps = eps
        self.is_weighted = is_volume_weighted
        self.reduction=reduction

    def forward(self, pred, gt, mask):
        
        # reshape according to the spatial dimention - to get n-dimensional unit vectors
        gt_vector = gt.reshape(gt.shape[:2] + (-1,)) * (1-self.eps)
        pred_vector = pred.reshape(gt.shape[:2] + (-1,)) * (1-self.eps)
        binary_mask_vector = mask.reshape(mask.shape[:1] + (-1,))       

        #clip cosinus to -1,1 for numerical stability
        angle_errors = torch.acos(torch.clip(torch.sum(gt_vector*pred_vector, dim=1, keepdim=False),-1,1))
        if not self.is_weighted:
            loss = torch.sum(angle_errors*angle_errors*binary_mask_vector, dim=1)
        else:
            #inverse square root of area
            weight = 1/torch.sqrt(binary_mask_vector.sum())
            loss = torch.sum(angle_errors*angle_errors*binary_mask_vector*weight, dim=1)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss
