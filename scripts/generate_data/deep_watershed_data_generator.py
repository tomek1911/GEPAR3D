import os
import random
import json
import re
import torch
import nibabel as nib
import torch.nn.functional as f
from collections import defaultdict, Counter
from tqdm import tqdm
from torch.utils.data import SequentialSampler, RandomSampler, WeightedRandomSampler
from sobel_filter import SobelFilter

import numpy as np
from scipy.ndimage import distance_transform_edt as edt_cpu

try:
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as edt_gpu
except ImportError:
    cp = None
    edt_gpu = None

### DATA PREPARATION FUNCTIONS ###

def load_dataset_json(json_path):
    with open(json_path, 'r') as f:
        dataset_info = json.load(f)
    return dataset_info.get('labels', {})

### Distance and Direction Maps Generation Functions
 
class WatershedDistanceMapGenerator:
    def __init__(self, device="cpu"):
        """
        device : 'cpu' or 'cuda'
        """
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        if device == "cuda" and cp is None:
            raise RuntimeError("CuPy is not installed but device='cuda' was requested")
        
        self.device = device
        if device == "cuda":
            self.xp = cp
            self.edt = edt_gpu
        else:
            self.xp = np
            self.edt = edt_cpu
        
        self.sobel3d = SobelFilter(spatial_size=3)
        if self.device == "cuda":
            self.sobel3d = self.sobel3d.cuda()
            

    def _get_bbox(self, mask):
        """Return tight bbox slices for non-zero region (mask is xp array)."""
        xp = self.xp
        coords = xp.where(mask)
        if coords[0].size == 0:
            return None
        min_coords = [int(c.min()) for c in coords]
        max_coords = [int(c.max()) + 1 for c in coords]
        return tuple(slice(mi, ma) for mi, ma in zip(min_coords, max_coords))
    
    def get_edt_map(self, label_volume, min_area=500, m=1):
        xp = self.xp
        edt = self.edt
        unique_labels = np.unique(label_volume)
        unique_labels = unique_labels[unique_labels != 0]  # skip background
                
        labels = xp.asarray(label_volume)
        distance_map = xp.zeros_like(labels, dtype=xp.float32)
        
        # ensure labels on correct device
        labels = xp.asarray(label_volume)
        distance_map = xp.zeros_like(labels, dtype=xp.float32)

        for lbl in unique_labels:
            lbl_xp = xp.asarray(lbl)

            class_mask = (labels == lbl_xp)
            voxel_count = int(xp.count_nonzero(class_mask))
            if voxel_count < min_area:
                continue

            bbox = self._get_bbox(class_mask)
            if bbox is None:
                continue

            #tight bbox
            mask_cropped = class_mask[bbox]         
            s0, s1, s2 = mask_cropped.shape

            # create padded array of zeros and place mask in its center
            padded_shape = (s0 + 2*m, s1 + 2*m, s2 + 2*m)
            mask_padded = xp.zeros(padded_shape, dtype=mask_cropped.dtype)

            # place cropped mask at offset m in each axis
            mask_padded[m:m+s0, m:m+s1, m:m+s2] = mask_cropped

            # compute EDT on padded mask
            edt_padded = edt(mask_padded, sampling=(1, 1, 1))

            # normalize per-instance using full padded edt
            max_dist = float(edt_padded.max()) if edt_padded.size else 0.0
            if max_dist > 0:
                edt_padded = edt_padded / max_dist

            # crop center back to original bbox size
            cropped_edt = edt_padded[m:m+s0, m:m+s1, m:m+s2]  # shape == (s0,s1,s2)

            # assign only where object exists in original bbox
            mask_local = mask_cropped
            if xp.count_nonzero(mask_local) == 0:
                continue

            local_indices = xp.where(mask_local)  # indices inside bbox
            # convert local indices to global indices in distance_map
            global_indices = tuple(local_indices[dim] + bbox[dim].start for dim in range(3))

            # assign values from cropped_edt[local_indices] into distance_map[global_indices]
            distance_map[global_indices] = cropped_edt[local_indices]

        distance_map[distance_map < 1e-2] = 0

        if self.device == "cuda":
            distance_map = xp.asnumpy(distance_map)

        return distance_map
    
    def get_dir_map(self, distance_map):
        if self.device == "cuda":
            distance_map = torch.from_numpy(distance_map).to('cuda')
        else:
            distance_map = torch.from_numpy(distance_map)
        distance_map = distance_map.view(1, 1, *distance_map.shape)
        direction_map = self.sobel3d(distance_map)
        direction_map = f.normalize(direction_map, p=2.0, dim=0, eps=1e-8)
        
        if self.device == "cuda":
            direction_map = direction_map.squeeze().cpu().numpy()
        else:   
            direction_map = direction_map.squeeze().numpy()
        return direction_map

def generate_watershed_maps(input_data_list, output_dir, device='cuda'):
    """
    Generate direction maps for each item in data_list.
    Each item is a dictionary with 'image' and 'label' keys.
    """
    distance_map_gen = WatershedDistanceMapGenerator(device=device)      
    for item in tqdm(input_data_list):
        nib_vol = nib.load(item)
        label_volume = np.asanyarray(nib_vol.dataobj, dtype=np.uint8) 
           
        dist_map =  distance_map_gen.get_edt_map(label_volume)
        dir_map = distance_map_gen.get_dir_map(dist_map)
        dir_map = np.moveaxis(dir_map, 0, -1) # make it channel last
        
        # Save
        file_name = item.split('/')[-1].replace('_primary.nii.gz', '.nii.gz')
        nib.save(nib.Nifti1Image(dist_map, nib_vol.affine), os.path.join(output_dir, 'distance_maps', file_name))
        nib.save(nib.Nifti1Image(dir_map, nib_vol.affine), os.path.join(output_dir, 'direction_maps', file_name))
        
        # DIST_DIR
        dist_map_ch = dist_map[..., np.newaxis]           # shape (X, Y, Z, 1)
        combined_maps = np.concatenate([dist_map_ch, dir_map], axis=-1)  # shape: (X, Y, Z, 4)
        # Use the header of the first image, but adjust dimensionality
        hdr = nib_vol.header.copy()
        hdr.set_data_dtype(np.float32)
        hdr.set_data_shape(combined_maps.shape)
        hdr['dim'][0] = 4                       # number of dimensions
        hdr['dim'][1] = combined_maps.shape[0]  # X
        hdr['dim'][2] = combined_maps.shape[1]  # Y
        hdr['dim'][3] = combined_maps.shape[2]  # Z
        hdr['dim'][4] = combined_maps.shape[3]  # channels (C)
        hdr['intent_code'] = 1007      # NIFTI_INTENT_VECTOR
        hdr['intent_name'] = b'Vector'
        
        # Save
        nib.save(nib.Nifti1Image(combined_maps, affine=nib_vol.affine, header=hdr), os.path.join(output_dir, 'distdir_maps', file_name))
        
if __name__ == "__main__":
    import nibabel as nib
    from tqdm import tqdm
    
    label_paths = ['data/...'] # TODO add you data paths here  
    for label_path in tqdm(label_paths):
        # Load NIfTI
        nib_vol = nib.load(label_path)
        label_volume = nib_vol.get_fdata().astype(np.int32)
        pixdim = nib_vol.header.get_zooms()[0]
        
        distance_map_gen = WatershedDistanceMapGenerator(device='cuda')
        
        print(f"Processing: {label_path}")
        dist_map =  distance_map_gen.get_edt_map(label_volume)
        dir_map = distance_map_gen.get_dir_map(dist_map)
        
        out_img = nib.Nifti1Image(dist_map, nib_vol.affine, nib_vol.header)
        nib.save(out_img, f"data/deep_watershed_volumes/distance_maps/{label_path.split('/')[-1]}")
        
        out_img = nib.Nifti1Image(dir_map, nib_vol.affine, nib_vol.header)
        nib.save(out_img, f"data/deep_watershed_volumes/direction_maps/{label_path.split('/')[-1]}")