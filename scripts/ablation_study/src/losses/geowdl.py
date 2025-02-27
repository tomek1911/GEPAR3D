import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from monai.networks.utils import one_hot

class GeometricWassersteinDiceLoss(_Loss):
    def __init__(self, dist_matrix, reduction='mean'):
        '''
        References:
        ===========
        [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
            Segmentation using Holistic Convolutional Networks."
            Fidon L. et al. MICCAI BrainLes (2017).
        [2] "Generalised dice overlap as a deep learning loss function
            for highly unbalanced segmentations."
            Sudre C., et al. MICCAI DLMIA (2017).
        [3] "Comparative study of deep learning methods for the automatic
            segmentation of lung, lesion and lesion type in CT scans of
            COVID-19 patients."
            Tilborghs, S. et al. arXiv preprint arXiv:2007.15546 (2020).
        '''
        super(GeometricWassersteinDiceLoss, self).__init__(reduction=reduction)
        
        self.M = dist_matrix
        if isinstance(self.M, np.ndarray):
            self.M = torch.from_numpy(self.M)
            
        self.num_classes = self.M.size(0)
        self.reduction = reduction
        self.smooth_nr = 1e-6
        self.smooth_dr = 1e-6

        
    def weight_true_positives(self, alpha, flat_target, wasserstein_distance_map):
        alpha = alpha.to(flat_target.device)
        alpha_extended = torch.gather(
            alpha.unsqueeze(2).expand(-1, -1, flat_target.size(1)),
            index=flat_target.unsqueeze(1),
            dim=1
        ).squeeze(1)
        return torch.sum(alpha_extended * (1. - wasserstein_distance_map), dim=1)


    def forward(self, input, target):
        """
        Compute the Wasserstein Dice loss between input and target tensors.
        :param input: tensor. input is the scores maps (before softmax).
        The expected shape of input is (N, C, H, W, D) in 3d
        and (N, C, H, W) in 2d.
        :param target: target is the target segmentation.
        The expected shape of target is (N, H, W, D) or (N, 1, H, W, D) in 3d
        and (N, H, W) or (N, 1, H, W) in 2d.
        :return: scalar tensor. Loss function value.
        """
        target = target.long()
        
        flat_input = input.view(input.size(0), input.size(1), -1)  # b,c,s
        flat_target = target.view(target.size(0), -1)  # b,s
        probs = F.softmax(flat_input, dim=1)  # b,c,s
        wass_dist_map = self.wasserstein_distance_map(probs, flat_target)
        
        alpha = torch.ones(flat_target.size(0), self.num_classes, device=flat_target.device, dtype=torch.float)
        alpha[:, 0] = 0
        true_pos = self.compute_generalized_true_positive(alpha, flat_target, wass_dist_map)
        all_error = torch.sum(wass_dist_map, dim=1)
        denom = 2 * true_pos + all_error
        
        #WASSERSTEIN DICE
        wass_dice: torch.Tensor = (2.0 * true_pos + self.smooth_nr) / (denom + self.smooth_dr)
        wass_dice_loss : torch.Tensor = 1.0 - wass_dice.mean()

        return wass_dice_loss

    def wasserstein_distance_map(self, flat_proba, flat_target):
        """
        Compute the voxel-wise Wasserstein distance (eq. 6 in [1]) for
        the flattened prediction and the flattened labels (ground_truth)
        with respect to the distance matrix on the label space M.
        References:
        ===========
        [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
        Segmentation using Holistic Convolutional Networks",
        Fidon L. et al. MICCAI BrainLes 2017
        """
        # Turn the distance matrix to a map of identical matrix
        M_extended = torch.clone(self.M).to(flat_proba.device)
        M_extended = torch.unsqueeze(M_extended, dim=0)  # C,C -> 1,C,C
        M_extended = torch.unsqueeze(M_extended, dim=3)  # 1,C,C -> 1,C,C,1
        M_extended = M_extended.expand((
            flat_proba.size(0),
            M_extended.size(1),
            M_extended.size(2),
            flat_proba.size(2)
        ))

        # Expand the feature dimensions of the target
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)  # b,s -> b,1,s
        flat_target_extended = flat_target_extended.expand(  # b,1,s -> b,C,s
            (flat_target.size(0), M_extended.size(1), flat_target.size(1))
        )
        flat_target_extended = torch.unsqueeze(flat_target_extended, dim=1)  # b,C,s -> b,1,C,s

        # Extract the vector of class distances for the ground-truth label at each voxel
        M_extended = torch.gather(M_extended, dim=1, index=flat_target_extended)  # b,C,C,s -> b,1,C,s
        M_extended = torch.squeeze(M_extended, dim=1)  # b,1,C,s -> b,C,s

        # Compute the wasserstein distance map
        wasserstein_map = M_extended * flat_proba

        # Sum over the classes
        wasserstein_map = torch.sum(wasserstein_map, dim=1)  # b,C,s -> b,s
        return wasserstein_map

    def compute_generalized_true_positive(self, alpha, flat_target, wasserstein_distance_map):
        # Extend alpha to a map and select value at each voxel according to flat_target
        alpha_extended = torch.unsqueeze(alpha, dim=2)  # b,C -> b,C,1
        alpha_extended = alpha_extended.expand(  # b,C,1 -> b,C,s
            (flat_target.size(0), self.num_classes, flat_target.size(1))
        )
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)  # b,s -> b,1,s
        alpha_extended = torch.gather(
            alpha_extended, index=flat_target_extended, dim=1)  # b,C,s -> b,1,s
        alpha_extended = torch.squeeze(alpha_extended, dim=1) # b,1,s -> b,s

        generalized_true_pos = torch.sum(
            alpha_extended * (1. - wasserstein_distance_map),
            dim=1,
        )

        return generalized_true_pos