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
    CopyItemsD,
    EnsureTypeD,
    LambdaD,
    LoadImageD,
    OrientationD,
    RandAdjustContrastD,
    RandLambdaD,
    RandRotateD,
    RandScaleIntensityD,
    RandShiftIntensityD,
    RandSpatialCropSamplesD,
    ResizeWithPadOrCropD,
    ScaleIntensityRangeD,
    SpacingD,
    ThresholdIntensityD,
    ToDeviceD,
)
from src.custom_augmentation import CropForegroundFixedD
from monai.utils.type_conversion import convert_data_type
 
class Transforms():
    def __init__(self,
                 args,
                 device : str = 'cpu' 
                ) -> None:

        self.pixdim = (args.pixdim,)*3
        self.class_treshold = args.classes if args.classes == 1 else args.classes-1
        mode_interpolation_dict = {"image":   "bilinear",
                                    "label":   "nearest",
                                    "binary_label":   "nearest",
                                    "edt":     "bilinear",
                                    "edt_dir": "bilinear",
                                    "seed":    "bilinear"
                                    }

        if args.classes == 1:
            keys = args.keys.copy()
            # TRAIN DATASET - LOAD, PREPROCESSING, AUGMENTATIONS
            load_data = [
                    LoadImageD(keys=keys, reader='NibabelReader'),
                    EnsureChannelFirstD(keys=keys, channel_dim='no_channel'),
                    OrientationD(keys=keys, axcodes="RAS"),
                    ToDeviceD(keys=keys, device=device),
                    EnsureTypeD(keys=keys, data_type="tensor", device=device),
                    # GEOMETRIC - NON-RANDOM - PREPROCESING
                    SpacingD(keys=keys, pixdim=self.pixdim,
                            mode=("bilinear", "nearest"))]
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
                ThresholdIntensityD(
                keys="label", above=False, threshold=self.class_treshold, cval=self.class_treshold),
                ThresholdIntensityD(
                    keys="label", above=True, threshold=0, cval=0),
                ToDeviceD(keys=keys, device=device)]
            
            #augmentations
            random_augmentations = [
                # GEOMETRIC
                RandSpatialCropSamplesD(keys=keys, roi_size=args.patch_size,
                                        random_center=True, random_size=False, num_samples=args.crop_samples),
                ResizeWithPadOrCropD(keys=keys, spatial_size=args.padding_size, method = 'symmetric', mode='constant', constant_values=0),
                RandRotateD(keys=keys, range_x=args.rotation_range, range_y=args.rotation_range, range_z=args.rotation_range, mode=("bilinear", "nearest"), prob=0.5),
                # INTENSITY
                RandAdjustContrastD(keys="image",
                                    gamma=(0.5, 2.0),
                                    prob=0.25),
                RandShiftIntensityD(keys="image", offsets=0.20, prob=0.5),
                RandScaleIntensityD(keys="image", factors=0.15, prob=0.5),
                EnsureTypeD(keys=keys, data_type="tensor", device=device)]
            
            self.train_transform = Compose(load_data + intensity_preproc + labels_preproc + random_augmentations)
            
            # VAL DATASET - LOAD, PREPROCESSING, AUGMENTATIONS
            load_data = [
                LoadImageD(keys=keys, reader='NibabelReader'),
                EnsureChannelFirstD(keys=keys, channel_dim='no_channel'),
                OrientationD(keys=keys, axcodes="RAS"),
                ToDeviceD(keys=keys, device=device),
                EnsureTypeD(keys=keys, data_type="tensor"),
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
                # clip to number of classes - clip value equall to max class value
                ThresholdIntensityD(
                    keys=["label"], above=False, threshold=self.class_treshold, cval=self.class_treshold),
                # clip from below to 0 - all values smaler than 0 are replaces with zeros
                ThresholdIntensityD(
                    keys=["label"], above=True, threshold=0, cval=0),
                EnsureTypeD(keys=keys, data_type="tensor", device=device)
                ]
            
            self.val_transform = Compose(load_data + intensity_preproc + labels_preproc)
            self.test_transform = self.val_transform
            if args.seed != -1:
                state = np.random.RandomState(seed=args.seed)
                self.train_transform.set_random_state(seed=args.seed, state=state)
        else:
            #ablation config
            keys = args.keys.copy()
            if 'SEED' in args.configuration_name:
                keys = keys[:5]
            elif 'DIR' in args.configuration_name:
                keys = keys[:4]
            elif 'EDT' in args.configuration_name:
                keys = keys[:3]
            else:
                keys = keys[:2]
            # TRAIN DATASET - LOAD, PREPROCESSING, AUGMENTATIONS
            load_data = [LoadImageD(keys=keys, reader='NibabelReader'),
                         EnsureChannelFirstD(keys=[i for i in keys if i != 'edt_dir'], channel_dim='no_channel'), # edt_dir is 3-channel 
                         # AddChannelD(keys=keys[:-1]),
                         EnsureTypeD(keys=keys, data_type="tensor", device=device, track_meta=True),
                         OrientationD(keys=keys, axcodes="RAS"),     
                         # GEOMETRIC - NON-RANDOM - PREPROCESING
                         SpacingD(keys=keys, pixdim=self.pixdim, mode=[mode_interpolation_dict[key] for key in keys])
                         ]
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
                ThresholdIntensityD(keys="label", above=False, threshold=32, cval=32),
                # clip from below to 0 - all values smaler than 0 are replaces with zeros
                ThresholdIntensityD(keys="label", above=True, threshold=0, cval=0),
            ]
            if "MASK" in args.configuration_name:
                labels_preproc +=[
                CopyItemsD(keys="label", names="binary_label"),
                ThresholdIntensityD(keys="binary_label", above=False, threshold=1, cval=1)]
                keys.append("binary_label")
            # augmentations
            random_augmentations = [
                # use proper device
                RandLambdaD(keys=keys, prob=1.0, func=lambda x: x.to(device)),
                # Geometric
                RandSpatialCropSamplesD(keys=keys, roi_size=args.patch_size, random_center=True, random_size=False, num_samples=args.crop_samples),
                RandRotateD(keys=keys, range_x=args.rotation_range, range_y=args.rotation_range, range_z=args.rotation_range, mode=[mode_interpolation_dict[key] for key in keys], prob=0.5),
                # INTENSITY - RANDOM - DATA AUGMENTATION
                RandAdjustContrastD(keys="image", gamma=(0.5, 2.0), prob=0.25),
                RandShiftIntensityD(keys="image", offsets=0.20, prob=0.5),
                RandScaleIntensityD(keys="image", factors=0.15, prob=0.5)
            ]
            self.train_transform = Compose(load_data + intensity_preproc + labels_preproc + random_augmentations, lazy=args.lazy_interpolation)
            if args.seed != -1:
                state = np.random.RandomState(seed=args.seed)
                self.train_transform.set_random_state(seed=args.seed, state=state)
            if 'binary_label' in keys:
                keys.remove('binary_label')
            
            # VAL DATASET - LOAD, PREPROCESSING, AUGMENTATIONS
            load_data = [LoadImageD(keys=keys, reader='NibabelReader'),
                         EnsureChannelFirstD(
                             keys=[i for i in keys if i != 'edt_dir'], channel_dim='no_channel'),
                         OrientationD(keys=keys, axcodes="RAS"),
                         EnsureTypeD(keys=keys, data_type="tensor", device=device),
                         # GEOMETRIC - NON-RANDOM - PREPROCESING
                         SpacingD(keys=keys, pixdim=self.pixdim, mode=[mode_interpolation_dict[key] for key in keys]),
                        ]
            
            if args.crop_foreground:
                cropforeground_image = CropForegroundFixedD(keys="image",
                            source_key="label",
                            select_fn=lambda x: x > 0,
                            margin=args.spatial_crop_margin,
                            spatial_size=args.spatial_crop_size,
                            mode='constant',
                            return_coords=False,
                            value=-1000) #padding value -1000 for hounsfield air
                cropforeground_others = CropForegroundFixedD(keys=keys[1:],
                            source_key="label",
                            select_fn=lambda x: x > 0,
                            margin=args.spatial_crop_margin,
                            spatial_size=args.spatial_crop_size,
                            mode='constant',
                            return_coords=False,
                            value=0,
                            allow_missing_keys=True)
                load_data += [cropforeground_image, cropforeground_others]
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
                ThresholdIntensityD(keys=["label"], above=True, threshold=0, cval=0)]
            if "MASK" in args.configuration_name:
                labels_preproc +=[
                CopyItemsD(keys="label", names="binary_label"),
                ThresholdIntensityD(keys="binary_label", above=False, threshold=1, cval=1),
                RandLambdaD(keys=keys + ["binary_label"], prob=1.0, func=lambda x: x.to(device))]
                keys.append("binary_label")
            else:
                labels_preproc +=[RandLambdaD(keys=keys, prob=1.0, func=lambda x: x.to(device))]
            self.val_transform = Compose(load_data + intensity_preproc + labels_preproc)
           
           #TEST SET
          
            keys = ["image", "label"]
            #load file
            load_data = [LoadImageD(keys=keys, reader='NibabelReader'),
                         EnsureChannelFirstD(
                             keys=keys, channel_dim='no_channel'),
                         OrientationD(keys=keys, axcodes="RAS"),
                         ToDeviceD(keys=keys, device=device),
                         SpacingD(keys=keys, pixdim=self.pixdim,
                                  mode=("bilinear", "nearest")),
                         EnsureTypeD(keys=keys, data_type="tensor")]
            #preproc
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
            self.test_transform = Compose(load_data + intensity_preproc + labels_preproc)

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
