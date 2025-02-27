import torch
import numpy as np
from monai.transforms import (
    AsDiscrete,
    Activations,
    Compose,
    ToDevice
)
from monai.transforms import (
    EnsureChannelFirstD,
    # AddChannelD,
    CopyItemsD,
    EnsureTypeD,
    LambdaD,
    LoadImageD,
    OrientationD,
    ResizeD,
    RandAdjustContrastD,
    RandAffineD,
    RandRotateD,
    RandSpatialCropD,
    RandScaleIntensityD,
    RandShiftIntensityD,
    RandSpatialCropSamplesD,
    RandZoomD,
    RandFlipD,
    RandLambdaD,
    RandRotate90D,
    ResizeWithPadOrCropD,
    ScaleIntensityRangeD,
    SpacingD,
    ThresholdIntensityD,
    ToDeviceD,
)
from src.custom_augmentation import CropForegroundFixedD
 
class Transforms():
    def __init__(self,
                 args,
                 device : str = 'cpu' 
                ) -> None:

        self.pixdim = (args.pixdim,)*3
        self.class_treshold = args.classes if args.classes == 1 else args.classes-1
        keys = args.keys.copy()
        
        # TRAIN DATASET - LOAD, PREPROCESSING, AUGMENTATIONS
        load_data = [LoadImageD(keys=keys, reader='NibabelReader'),
                    EnsureChannelFirstD(keys=keys, channel_dim='no_channel'),
                    # AddChannelD(keys=keys[:-1]),
                    OrientationD(keys=keys, axcodes="RAS"),
                    ToDeviceD(keys=keys, device=device),
                    EnsureTypeD(keys=keys, data_type="tensor",
                                device=device),
                    # GEOMETRIC - NON-RANDOM - PREPROCESING
                    SpacingD(keys=keys, pixdim=self.pixdim, mode=("bilinear", "nearest"))]
        if args.crop_foreground:
            cropforeground = CropForegroundFixedD(keys=keys,
                        source_key="label",
                        select_fn=lambda x: x > 0,
                        margin=args.spatial_crop_margin,
                        spatial_size=args.spatial_crop_size,
                        mode='constant',
                        return_coords=False,
                        constant_values=(-1000, 0))
            load_data += [cropforeground]
        #preproc
        if args.z_score_norm:
            intensity_preproc = []
        else:
            intensity_preproc = [ScaleIntensityRangeD(keys="image",
                                                        a_min=0,
                                                        a_max=args.houndsfield_clip,
                                                        b_min=0.0,
                                                        b_max=1.0,
                                                        clip=True)]
        labels_preproc = [
            # clip to number of classes - clip value equal to max class value
            ThresholdIntensityD(keys="label", above=False, threshold=32, cval=32),
            # clip from below to 0 - all values smaler than 0 are replaces with zeros
            ThresholdIntensityD(keys="label", above=True, threshold=0, cval=0),
        ]
        
        # augmentations
        random_augmentations = [
            RandLambdaD(keys=keys, prob=1.0, func=lambda x: x.to(device)),
            ToDeviceD(keys=keys, device=device),
            # Geometric
            RandSpatialCropSamplesD(keys=keys, roi_size=args.patch_size,
                                    random_center=True, random_size=False, num_samples=args.crop_samples),
            RandRotateD(keys=keys, range_x=args.rotation_range, range_y=args.rotation_range, range_z=args.rotation_range, mode=("bilinear", "nearest"), prob=0.5),
            # INTENSITY - RANDOM - DATA AUGMENTATION
            RandAdjustContrastD(keys="image",
                                gamma=(0.5, 2.0),
                                prob=0.25),
            RandShiftIntensityD(keys="image", offsets=0.20, prob=0.5),
            RandScaleIntensityD(keys="image", factors=0.15, prob=0.5),
            # FINAL CHECK
            EnsureTypeD(keys=keys, data_type="tensor", device=device)
        ]
        self.train_transform = Compose(load_data + intensity_preproc + labels_preproc + random_augmentations)
        if args.seed != -1:
            state = np.random.RandomState(seed=args.seed)
            self.train_transform.set_random_state(seed=args.seed, state=state)
        # VAL DATASET - LOAD, PREPROCESSING, AUGMENTATIONS
        keys = args.keys
        load_data = [LoadImageD(keys=keys, reader='NibabelReader'),
                        EnsureChannelFirstD(keys=keys, channel_dim='no_channel'),
                        OrientationD(keys=keys, axcodes="RAS"),
                        ToDeviceD(keys=keys, device=device),
                        # GEOMETRIC - NON-RANDOM - PREPROCESING
                        SpacingD(keys=keys, pixdim=self.pixdim, mode=("bilinear", "nearest")),
                        EnsureTypeD(keys=keys, data_type="tensor")]
        
        if args.crop_foreground:
            cropforeground = CropForegroundFixedD(keys=keys,
                        source_key="label",
                        select_fn=lambda x: x > 0,
                        margin=args.spatial_crop_margin,
                        spatial_size=args.spatial_crop_size,
                        mode='constant',
                        return_coords=True,
                        constant_values=(-1000, 0))
            load_data += [cropforeground]
        #preproc
        if args.z_score_norm:
            intensity_preproc = []
        else:
            intensity_preproc = [ScaleIntensityRangeD(keys="image",
                                                        a_min=0,
                                                        a_max=args.houndsfield_clip,
                                                        b_min=0.0,
                                                        b_max=1.0,
                                                        clip=True)]
        labels_preproc = [
            # clip to number of classes - clip value equall to max class value
            ThresholdIntensityD(keys=["label"], above=False, threshold=32, cval=32),
            # clip from below to 0 - all values smaler than 0 are replaces with zeros
            ThresholdIntensityD(keys=["label"], above=True, threshold=0, cval=0),
            RandLambdaD(keys=keys, prob=1.0, func=lambda x: x.to(device))
        ]
        #validation transform
        self.val_transform = Compose(load_data + intensity_preproc + labels_preproc)
        #test transform - without initial crop - only original scan
        self.test_transform = Compose(load_data[:-1] + intensity_preproc + labels_preproc)
        self.binarize_transform = ThresholdIntensityD(keys="label", above=False, threshold=1, cval=1)

        if args.classes > 1:
            self.post_pred_train = Compose([Activations(softmax=True, dim=1),
                                      AsDiscrete(argmax=True,
                                                 dim=1,
                                                 keepdim=True)
                                    ])
            self.post_pred = Compose([Activations(softmax=True, dim=0),
                            AsDiscrete(argmax=True,
                                       dim=0,
                                       keepdim=True),
                            ToDevice(device=device)
                        ])
            self.post_pred_labels = Compose([AsDiscrete(argmax=False,
                                                        to_onehot=args.classes,
                                                        dim=0),
                                             ToDevice(device=device)
                                            ])
        elif args.classes == 1:
            self.post_pred = Compose([Activations(sigmoid=True),
                                      AsDiscrete(threshold=0.5)],
                                      ToDevice(device=device))

# GPU accelerated morphological dillation and erosion - cuda pytorch
def dilation2d(image: torch.tensor, kernel: torch.tensor = torch.ones((1, 1, 3, 3)), border_type: str = 'constant', border_value: int = 0, device: torch.device = torch.device('cpu')):
    # shape
    if len(kernel.shape) == 4:
        _, _, se_h, se_w = kernel.shape
    elif len(kernel.shape) == 3:
        _, se_h, se_w = kernel.shape
    # padding
    origin = [se_h // 2, se_w // 2]
    pad_margin = [origin[1], se_w - origin[1] -
                  1, origin[0], se_h - origin[0] - 1]
    volume_pad = torch.nn.functional.pad(
        image, pad_margin, mode=border_type, value=border_value)
    # kernel
    if kernel.device != device:
        kernel = kernel.to(device)
    # dilation
    out = torch.nn.functional.conv2d(
        volume_pad, kernel, padding=0).to(torch.int)
    dilation_out = torch.clamp(out, 0, 1)
    return dilation_out


def dilation3d(volume: torch.tensor, kernel: torch.tensor = torch.ones((1, 1, 3, 3, 3)), border_type: str = 'constant', border_value: int = 0, device: torch.device = torch.device('cpu')):
    # shape
    if len(kernel.shape) == 5:
        _, _, se_h, se_w, se_d = kernel.shape
    elif len(kernel.shape) == 4:
        _, se_h, se_w, se_d = kernel.shape
    # padding
    origin = [se_h // 2, se_w // 2, se_d // 2]
    pad_margin = [origin[2], se_d - origin[2] - 1, origin[1],
                  se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
    volume_pad = torch.nn.functional.pad(
        volume, pad_margin, mode=border_type, value=border_value)
    # kernel
    if kernel.device != device:
        kernel = kernel.to(device)
    # dilation
    out = torch.nn.functional.conv3d(volume_pad, kernel, padding=0)
    dilation_out = torch.clamp(out, 0, 1)
    return dilation_out


def erosion2d(image: torch.tensor, kernel: torch.tensor = torch.ones((1, 1, 3, 3)), border_type: str = 'constant', border_value: int = 0, device: torch.device = torch.device('cpu')):
    # shape
    if len(kernel.shape) == 4:
        _, _, se_h, se_w = kernel.shape
    elif len(kernel.shape) == 3:
        _, se_h, se_w = kernel.shape
    # padding
    origin = [se_h // 2, se_w // 2]
    pad_margin = [origin[1], se_w - origin[1] -
                  1, origin[0], se_h - origin[0] - 1]
    volume_pad = torch.nn.functional.pad(
        image, pad_margin, mode=border_type, value=border_value)
    # kernel
    if kernel.device != device:
        kernel = kernel.to(device)
    if torch.is_tensor(kernel):
        bias = -kernel.sum().unsqueeze(0) + 1
    else:
        bias = torch.tensor(-kernel.sum()).unsqueeze(0) + 1
    # erosion
    out = torch.nn.functional.conv2d(
        volume_pad, kernel, padding=0, bias=bias).to(torch.int)
    erosion_out = torch.add(torch.clamp(out, -1, 0), 1)
    return erosion_out


def erosion3d(volume: torch.tensor, kernel: torch.tensor = torch.ones((1, 1, 3, 3, 3)), border_type: str = 'constant', border_value: int = 0, device: torch.device = torch.device('cpu')):
    # shape
    if len(kernel.shape) == 5:
        _, _, se_h, se_w, se_d = kernel.shape
    elif len(kernel.shape) == 4:
        _, se_h, se_w, se_d = kernel.shape
    # padding
    origin = [se_h // 2, se_w // 2, se_d // 2]
    pad_margin = [origin[2], se_d - origin[2] - 1, origin[1],
                  se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
    volume_pad = torch.nn.functional.pad(
        volume, pad_margin, mode=border_type, value=border_value)
    # kernel
    if kernel.device != device:
        kernel = kernel.to(device)
    if torch.is_tensor(kernel):
        bias = -kernel.sum().unsqueeze(0) + 1
    else:
        bias = torch.tensor(-kernel.sum()).unsqueeze(0) + 1
    # erosion
    out = torch.nn.functional.conv3d(
        volume_pad, kernel, padding=0, stride=1, bias=bias)
    erosion_out = torch.add(torch.clamp(out, -1, 0), 1)
    return erosion_out
