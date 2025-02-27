from typing import Optional
import numpy as np
from skimage.segmentation import watershed
from skimage.measure import label, regionprops

def deep_watershed_with_voting(deep_watershed_basins: np.array, multiclass_segmentation: np.array, binary_mask: Optional[np.array] = None,
                               markers: Optional[np.array] = None, instance_ground_truth: Optional[np.array] = None,
                               seed_distance_treshold: float = 0.5, calculate_metrics: bool = False):
    
    #check required inputs for deep watershed algorithm
    if markers is None:
        markers = deep_watershed_basins.copy()
        markers = np.where(markers > seed_distance_treshold, 1, 0)
    if binary_mask is None:
        binary_mask = multiclass_segmentation.copy()
        binary_mask = np.where(binary_mask >= 1, 1, 0)
    
    #label seeds 
    instances = label(markers, connectivity=3, return_num=False)

    #apply watershed algorithm based on negative distance using instances as seeds and masked by binary segmentation
    instance_masks = watershed(-deep_watershed_basins, instances, mask=binary_mask)

    #get instance masks and bboxes (regionprops) and perform majority voting to assign classes to instances
    output = np.zeros_like(instance_masks)
    instance_masks_props = regionprops(instance_masks)

    voting_instances = [] 
    pred_instances = []
    gt_instances = []
    
    #MAJORITY VOTING
    for idx, i in enumerate(instance_masks_props):        
        #get tooth instance voxels based on region of interest fr                                                                                                                                                                      om original model output
        pred_instance  = multiclass_segmentation[i.bbox[0]:i.bbox[3], i.bbox[1]:i.bbox[4], i.bbox[2]:i.bbox[5]][i.image].astype(np.int8)
        #get class with the most votes (ignore background class)
        votes_pred = np.bincount(pred_instance)
        
        majority_class_pred = 0
        if len(votes_pred)>1:
            majority_class_pred = np.argmax(votes_pred[1:])+1

        #relabel instance voxels based on winner
        voting_instances.append(majority_class_pred)
        output[i.bbox[0]:i.bbox[3], i.bbox[1]:i.bbox[4], i.bbox[2]:i.bbox[5]][i.image] = majority_class_pred

    #CALCULATE METRICS 
    if calculate_metrics and instance_ground_truth is not None:
        gt_masks_props = regionprops(instance_ground_truth.astype(np.int8))
        for idx, i in enumerate(gt_masks_props):    
            #GT    
            gt_instance = instance_ground_truth[i.bbox[0]:i.bbox[3], i.bbox[1]:i.bbox[4], i.bbox[2]:i.bbox[5]][i.image].astype(np.int8)
            votes = np.bincount(gt_instance)
            gt_instance_class = np.argmax(votes[1:])+1
            #PRED
            if gt_instance_class in voting_instances:
                predicted_instance = output[i.bbox[0]:i.bbox[3], i.bbox[1]:i.bbox[4], i.bbox[2]:i.bbox[5]][i.image].astype(np.int8)
                votes_pred = np.bincount(predicted_instance)
                pred_instance_class = np.argmax(votes_pred[1:])+1
                pred_instances.append(pred_instance_class)
                gt_instances.append(gt_instance_class)
                
        return output, [pred_instances, gt_instances]   
    return output, None