import torch
import numpy as np
from monai.transforms import AsDiscrete, Compose, Activations
from monai.inferers import sliding_window_inference
from src.models.unet import UNet
from src.data_augmentation import dilation3d, erosion3d
from skimage import morphology

def morphological_closing3d(input_tensor : torch.tensor, e_k : int, d_k : int, device : torch.device = torch.device("cpu")):
    if device.type == "cpu":
        d_kernel = torch.ones((1,1)+(d_k,)*3)
        e_kernel = torch.ones((1,1)+(e_k,)*3)
    elif device.type == "cuda":
        d_kernel = torch.cuda.FloatTensor(*(1,1)+(d_k,)*3, device=device).fill_(1)
        e_kernel = torch.cuda.FloatTensor(*(1,1)+(e_k,)*3, device=device).fill_(1)
    out = dilation3d(erosion3d(input_tensor, e_kernel, device=device), d_kernel, device=device)
    return out

def get_bounding_box(volume : np.array, margins : tuple = (0.03, 0.03, 0.06), class_treshold : int=1):
    # get implicit edges of tooths according to treshold
    coords = np.argwhere(volume >= class_treshold)
    min_slice = coords.min(axis=0)
    max_slice = coords.max(axis=0)
    #update roi edges with margins - margins are required because roots area may be lost
    new_shape = max_slice-min_slice
    crop_margins = tuple(int(r*x) for x, r in zip(new_shape, margins))
    # print(f"margins: {crop_margins}")
    min_slice -= crop_margins
    max_slice += crop_margins
    #limit to boundaries of original shape - ROI cannot be larger than original scan
    min_slice[min_slice < 0] = 0
    max_slice = tuple(map(lambda a, da: min(a,da), tuple(max_slice), volume.shape))
    #slices to cut numpy array and roi_bounds to draw in pyvista
    slices = tuple(map(slice, min_slice, max_slice))
    roi_bounds = np.array([item for subitem in list(zip(min_slice, max_slice)) for item in subitem]) 
    return slices, roi_bounds

def load_coarse_binary_model(args_binary, device):
    binary_checkpoint_path = 'checkpoints/coarse/determined_valance_1193_classes_1_Dice_CE/model-UNet-1class-fold-0_current_best_val.pt'
    feature_maps = tuple(2**i*args_binary.n_features for i in range(0, args_binary.unet_depth))
    strides = list((args_binary.unet_depth-1)*(2,))
    model_coarse = UNet(spatial_dims=3, in_channels=1, out_channels=args_binary.classes,
                channels=feature_maps, strides=strides, act='relu', norm='instance', bias=False)
    model_coarse.load_state_dict(torch.load(binary_checkpoint_path, map_location=device)['model_state_dict'], strict=True)
    model_coarse = model_coarse.to(device)
    model_coarse.eval()
    return model_coarse

def detect_roi(args_binary, data_sample, predictor, device, margins = (0.1,0.1,0.3)):
    output_logit = sliding_window_inference(data_sample["image"], roi_size=args_binary.patch_size, sw_batch_size=8, predictor=predictor, overlap=0.6, sw_device=device,
                                    device=device, mode='gaussian', sigma_scale=0.125, padding_mode='constant', cval=0, progress=False) #center padding with zeros(cval) if image smaller than 256,256,256 (patch_size)
    post_transform_binary = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    seg_pred = post_transform_binary(output_logit)
    seg_pred = morphological_closing3d(seg_pred, d_k=5, e_k=5, device=device)
    #remove_small_objects requires boolean array if only one class (for labeled use int)
    seg_pred = morphology.remove_small_objects(seg_pred.to(torch.bool).squeeze().cpu().numpy(), min_size=50, connectivity=1).astype(np.int32)
    slices, roi_bounds = get_bounding_box(seg_pred, margins=margins)
    tensor_slices = (slice(0,1,None),)*2+slices
    teeth_roi = data_sample["image"][tensor_slices]
    teeth_roi_label = data_sample["label"][tensor_slices]
    return teeth_roi, teeth_roi_label, tensor_slices, roi_bounds