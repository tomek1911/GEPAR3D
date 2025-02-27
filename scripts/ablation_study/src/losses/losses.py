# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import torch
import torch.nn as nn
import numpy as np
from monai.networks import one_hot
from monai.losses import FocalLoss

if __name__ == "__main__":
    from geowdl import GeometricWassersteinDiceLoss
else:
    from .geowdl import GeometricWassersteinDiceLoss

#Wasserstein distance matrix in probabilistic space based on tooth statistical shape model
def get_tooth_dist_matrix(device, config : str = "config_0", background_penalty=1.0, background_penalty_FP=0, background_penalty_FN=0, is_normalize=True):
    if config == 'random':
        dist_matrix = torch.from_numpy(np.load('scripts/ablation_study/src/losses/weights/wasserstein_random_matrix.npy'))
        print("Sanity check II - random matrix as wasserstein matrix")
    elif config == 'geom_prior':
        dist_matrix = torch.from_numpy(np.load('scripts/ablation_study/src/losses/weights/wasserstein_ssm_quad_w_add_matrix.npy'))
        print("Geometrical prioir with semantical penalties between quadrants. Inter-quadrant penalties 0, 0.1, 0.2, 0.3")
    
    #add background penalties, equal or different depending on type of error
    if background_penalty_FN > 0 and background_penalty_FP > 0:
        wasserstein_matrix = torch.concatenate([torch.ones((1,32))*background_penalty_FP, dist_matrix])
        wasserstein_matrix = torch.concatenate([torch.ones((33,1))*background_penalty_FN, wasserstein_matrix], axis=1)
    else:
        wasserstein_matrix = torch.concatenate([torch.ones((1,32))*background_penalty, dist_matrix])
        wasserstein_matrix = torch.concatenate([torch.ones((33,1))*background_penalty, wasserstein_matrix], axis=1)
    wasserstein_matrix[0][0]=0 #correctly classified background class
    
    if is_normalize:
        wasserstein_matrix /= wasserstein_matrix.max()
    #print whole
    np.set_printoptions(threshold=10000)
    print(np.round(wasserstein_matrix.numpy(),3))
    np.set_printoptions(threshold=1000) #restore default
    #print truncated
    print(np.round(wasserstein_matrix.numpy(),3))
    return wasserstein_matrix.to(device)

class GeometricalWassersteinDiceLoss(nn.Module):
    def __init__(self, cost_matrix, class_weights=None, classification_loss='cross_entropy', 
                 dice_loss_type='gwd', reduction='mean', to_onehot_y=True, gamma=2.0):
        super().__init__()
        self.classification_loss_name = classification_loss
        self.dice_loss_type = dice_loss_type
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)
        self.focal_loss = FocalLoss(include_background=True, to_onehot_y=to_onehot_y, gamma=gamma, weight=class_weights, reduction=reduction)
        self.geometricWassersteinDice = GeometricWassersteinDiceLoss(dist_matrix=cost_matrix, reduction=reduction)
        
    def ce(self, input: torch.Tensor, target: torch.Tensor):
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)
        return self.ce_loss(input, target)
    
    def focal(self, input: torch.Tensor, target: torch.Tensor):
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )
        if self.to_onehot_y:
            n_pred_ch = input.shape[1]
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)
        return self.focal_loss(input, target)
    
    def gwdl(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return self.geometricWassersteinDice(y_pred, y_true)
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if self.classification_loss_name == 'cross_entropy':
            cls_loss = self.ce(y_pred, y_true)
        elif self.classification_loss_name == 'focal_loss':
            cls_loss = self.focal(y_pred, y_true)
        if self.dice_loss_type=='gwd':
            seg_loss = self.gwdl(y_pred, y_true)
        return seg_loss, cls_loss