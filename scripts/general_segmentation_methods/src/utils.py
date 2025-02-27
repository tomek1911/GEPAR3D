import os
import torch
import numpy as np
import nibabel as nib
from monai.data import MetaTensor, decollate_batch
from monai.transforms import Orientationd

def label_rescale(image_label, w_ori, h_ori, z_ori):
    w_ori, h_ori, z_ori = int(w_ori), int(h_ori), int(z_ori)
    is_tensor_input = torch.is_tensor(image_label)
    if not is_tensor_input:
        image_label = torch.from_numpy(image_label)
    if image_label.device.type != 'cuda':
        image_label = image_label.cuda(0)
    teeth_ids = image_label.unique()
    image_label_ori = torch.zeros((image_label.shape[0], image_label.shape[1], w_ori, h_ori, z_ori)).to(image_label.device)
    for label_id in range(len(teeth_ids)):
        if label_id == 0:
            continue
        image_label_bn = (image_label == teeth_ids[label_id]).float()
        # image_label_bn = image_label_bn[:, :, :]
        image_label_bn = torch.nn.functional.interpolate(image_label_bn, size=(w_ori, h_ori, z_ori), mode='trilinear')
        # image_label_bn = image_label_bn[0, 0, :, :, :]
        image_label_ori[image_label_bn > 0.5] = teeth_ids[label_id]
    return image_label_ori
     

def save_nifti(array, path, filename, pixdim = 0.4, label_meta_dict=None, affine=None, dtype = np.int16):
    if label_meta_dict is None:
        affine = np.eye(4) * pixdim
        affine[3][3]=1.0
    else:
        if len(array.shape)==5:
            label_meta_dict = decollate_batch(label_meta_dict)[0]
            array = array[0]
        affine = label_meta_dict["affine"].numpy()
        space = label_meta_dict['space']
        if nib.aff2axcodes(affine) == ('L', 'P', 'S') and space == "RAS":
            t = MetaTensor(array, meta=label_meta_dict)
            array=Orientationd(keys="label", axcodes="RAS")({"label": t})["label"]
        # elif nib.aff2axcodes(affine) == ('R', 'A', 'S') and space == "LPS":
        #     t = MetaTensor(array, meta=label_meta_dict)
        #     array=Orientationd(keys="label", axcodes="LPS")({"label": t})["label"]
    if torch.is_tensor(array):
        nib_array = nib.Nifti1Image(array.cpu().squeeze().numpy().astype(dtype), affine=affine)
    else:
        nib_array = nib.Nifti1Image(array.astype(dtype), affine=affine)
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = os.path.join(path, filename)
    nib.save(nib_array, save_path)