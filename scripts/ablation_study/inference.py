import os
import yaml
import json
from argparse import Namespace

if os.getcwd().split('/')[-1] != 'CBCT_SEG_E2E':
    os.chdir('../')

#specific to experiment config
experiment_config_file = 'configs/experiments_config.yaml'
with open(experiment_config_file, 'r') as file:
    experiments_config = yaml.safe_load(file)

experiment_name = 'dentnet_ablation_study'
experiment_configurations = experiments_config[experiment_name]['experiments']
experiments_count = len(experiment_configurations)

#general config
general_config_file = 'configs/dentnet_geometric_prior_multitask.yaml'
with open(general_config_file, 'r') as file:
    general_config = yaml.safe_load(file)
general_config['args']['experiment_name'] = "geometric_prior_multitask"

#update general configuration
update_config = experiments_config[experiment_name]['update_config'].get('general', None)
if update_config is not None:
    general_config['args'].update(update_config)
    
#load public dataset data split
with open('configs/data_split.json', 'r') as f:
    d = json.load(f)

args = Namespace(**general_config['args'])

import glob
import numpy as np
import torch
import torch.nn as nn
import warnings

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from natsort import natsorted
from monai.data import DataLoader
from monai.data.dataset import PersistentDataset
from monai.inferers import sliding_window_inference
from monai.metrics import (HausdorffDistanceMetric, SurfaceDistanceMetric, DiceMetric)
from monai.networks.utils import one_hot
from monai.transforms import Compose
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    OrientationD,
    LoadImageD,
    ScaleIntensityRangeD,
    SpacingD,
    EnsureTypeD,
    ThresholdIntensityD,
    RandLambdaD,
    ToDeviceD
)
from monai.transforms import (
    Compose,
    AsDiscrete
)


from src.models.gepar3d import GEPAR3D
from src.commons.log_image import Logger
from src.watershed_ablation import deep_watershed_with_voting
from inference_utils import load_coarse_binary_model, detect_roi
from src.utils import save_nifti

#CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
torch.set_num_threads(2)

#TRANSFORM
#inference with preprocessing - load labels to calculate metrics
keys = ["image", 'label']
inference_transform_pre = Compose(
        [
        LoadImageD(keys=keys, reader='NibabelReader'),
        EnsureChannelFirstD(keys=keys, channel_dim='no_channel'),
        OrientationD(keys=keys, axcodes="RAS"),
        ToDeviceD(keys=keys, device=device),
        EnsureTypeD(keys=keys, data_type="tensor", device=device),
        SpacingD(keys=keys,
                    pixdim=(args.pixdim,)*3,
                    mode=("bilinear", 'nearest'),
                    recompute_affine=True),
        ScaleIntensityRangeD(
            keys=["image"], a_min=0, a_max=args.houndsfield_clip,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        ThresholdIntensityD(keys=["label"], above=False, threshold=32, cval=32), #clip to number of classes - clip value equall to max class value
        ThresholdIntensityD(keys=["label"], above=True, threshold=0, cval=0),
        RandLambdaD(keys=keys, prob=1.0, func=lambda x: x.to(device))
        ]
    )

#METRICS
reduction = 'mean_batch'
dice = DiceMetric(include_background=False, reduction=reduction)
hausdorf = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=None, get_not_nans=False, directed=False, reduction=reduction)
hausdorf95 = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=95, get_not_nans=False, directed=False, reduction=reduction)
asd = SurfaceDistanceMetric(include_background=False, distance_metric='euclidean', reduction=reduction, symmetric=True)

metrics_list = [dice, hausdorf, asd]
metrics_names_list = ['DSC', 'HD', 'ASSD']

classes=args.classes
patch_size=tuple(args.patch_size)

#DATA
dataset = "centers"

if dataset == "centers":
    data_dir = 'data/test_data'
    centers = 'ABCD' #AB - GEPAR3D dataset, C - Z.Cui et al. dataset, D - ToothFairy2
    nifti_paths_scans_A = natsorted(glob.glob(os.path.join(data_dir, 'scans', 'centerA', '*.nii.gz')))
    nifti_paths_labels_A = natsorted(glob.glob(os.path.join(data_dir, 'labels', 'centerA', '*.nii.gz')))
    nifti_paths_scans_B = natsorted(glob.glob(os.path.join(data_dir, 'scans', 'centerB', '*.nii.gz')))
    nifti_paths_labels_B = natsorted(glob.glob(os.path.join(data_dir, 'labels', 'centerB', '*.nii.gz')))
    nifti_paths_scans_C = natsorted(glob.glob(os.path.join(data_dir, 'scans', 'centerC', '*.nii.gz')))
    nifti_paths_labels_C = natsorted(glob.glob(os.path.join(data_dir, 'labels', 'centerC', '*.nii.gz')))
    nifti_paths_scans_D = natsorted(glob.glob(os.path.join(data_dir, 'scans', 'centerD', '*.nii.gz')))
    nifti_paths_labels_D = natsorted(glob.glob(os.path.join(data_dir, 'labels', 'centerD', '*.nii.gz')))
print(centers)
nifti_paths_scans = []
nifti_paths_labels = []

for center in centers:
    if center == 'A':
        nifti_paths_scans += nifti_paths_scans_A
        nifti_paths_labels += nifti_paths_labels_A
    elif center == 'B':
        nifti_paths_scans += nifti_paths_scans_B
        nifti_paths_labels += nifti_paths_labels_B
    elif center == 'C':
        nifti_paths_scans += nifti_paths_scans_C
        nifti_paths_labels += nifti_paths_labels_C
    elif center == 'D':
        nifti_paths_scans += nifti_paths_scans_D
        nifti_paths_labels += nifti_paths_labels_D
        
if nifti_paths_labels is None:
    nifti_list = [{'image' : scan} for scan in nifti_paths_scans]
else:
    nifti_list = [{'image' : scan, 'label': label} for scan, label in zip(nifti_paths_scans, nifti_paths_labels)]

clear_cache = True
if args.clear_cache:
    test_cache = glob.glob(os.path.join(os.path.join('data', 'cached_datasets', 'testset', '*.pt')))
    if len(test_cache) != 0:
        for file in test_cache:
            os.remove(file)
    print(f"Cleared testset cache: {len(test_cache)} files.")
    
test_dataset = PersistentDataset(nifti_list, inference_transform_pre, cache_dir=os.path.join('data', 'cached_datasets', 'testset'))
test_loader = DataLoader(test_dataset, num_workers=0, batch_size=1)

##  coarse binary roi detection model
# load config binary coarse unet
config_file = 'configs/dentnet_roi_binary_coarse_segmentation.yaml'
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)
args_binary = Namespace(**config['args'])
model_roi_binary = load_coarse_binary_model(args_binary, device)
    
#missing_teeth is 0
all_teeth = torch.arange(1,33).long()     
target_names = [f'tooth_{i}' for i in range(1,33)]

calculate_metrics = False
log3d_results = True
is_save_nifti = True
crop_roi = True 

experiment_folder = "gepar3d_ablation"
model_name = args.model_name
checkpoint_paths = sorted(glob.glob(f"checkpoints/{experiment_folder.split('_')[0]}/ablation_study{experiment_folder}/*"))
checkpoint_configurations = [('_').join(file.split('/')[-1].split('_')[3:]) for file in checkpoint_paths]

for configuration_name in experiment_configurations:
    if log3d_results:
        log = Logger(num_classes=33, is_log_3d=True)
        
    #experiment specific configuration 
    specific_config = experiments_config[experiment_name]['update_config'].get(configuration_name, None)
    if specific_config is not None:
        general_config['args'].update(specific_config)
    args = Namespace(**general_config['args'])
    
    checkpoint_id = [configuration_name==conf for conf in checkpoint_configurations].index(True)
    checkpoint_path = os.path.join(checkpoint_paths[checkpoint_id], 'classes_33_Dice_WCE', f'model-{model_name}-33class-fold-0_current_best_val.pt')
    
    state_dict=torch.load(checkpoint_path, map_location=device)
    
    print('\n------------------------')
    print(f"Inference of config/method: {configuration_name},")
    print(f"checkpoint path: {checkpoint_path},")
    print(f"model from epoch: {state_dict['epoch']},")
    print(f"train DSC: {state_dict.get('model_train_dice',0.0)*100:.2f},")
    print(f"train HD: {state_dict.get('model_train_hd', 0.0):.2f},")
    print(f"val DSC: {state_dict.get('model_val_dice', 0.0)*100:.2f},")
    print(f"val HD: {state_dict.get('model_val_hd', 0.0):.2f},")
    print(f"val multi-class DSC: {state_dict.get('val_dice_multiclass_agg', 0.0)*100:.2f}.", flush=True)
        
    all_metrics = []
    dice_results = []
    model = GEPAR3D(spatial_dims=3, in_channels=1, out_channels=classes, act='relu', norm='instance', bias=False, backbone_name='resnet34', inference_mode=True, configuration=configuration_name)
    model.load_state_dict(state_dict['model_state_dict'], strict=True)
    model = model.to(device)
    model.eval()

    print('inference in progress...')
    with torch.no_grad():
        for idx, test_data in enumerate(tqdm(test_loader, disable=True)):
            file_name = test_data['label'].meta['filename_or_obj'][0]
            image, label = test_data["image"], test_data["label"]

            #binary roi detection
            original_shape = image.shape
            if crop_roi and 'centerC' in file_name or 'centerD' in file_name:
                image, label, roi_slices, roi_bounds = detect_roi(args_binary, data_sample=test_data, predictor=model_roi_binary, device=device)
            (multiclass_segmentation, edt) = sliding_window_inference(image, roi_size=args.patch_size, sw_batch_size=8, predictor=model, overlap=0.6, sw_device=device,
                                                                      device=device, mode='gaussian', sigma_scale=0.125, padding_mode='constant', cval=0, progress=False) 
     
            if 'EDT' in configuration_name:
                #watershed algorithm inputs
                multiclass_segmentation = AsDiscrete(argmax=True)(multiclass_segmentation[0]).squeeze().cpu().numpy()
                edt = nn.Threshold(1e-3, 0)(edt).squeeze().cpu().numpy()
                #deep watershed
                pred_np, _ = deep_watershed_with_voting(edt, multiclass_segmentation)
                multiclass_segmentation_pred = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0).to(device)
                test_seg_pred_onehot = one_hot(multiclass_segmentation_pred, num_classes=args.classes, dim=1)
            else:
                multiclass_segmentation_pred = AsDiscrete(argmax=True, dim=1, keepdim=True)(multiclass_segmentation)
                test_seg_pred_onehot = one_hot(multiclass_segmentation_pred, num_classes=args.classes, dim=1)
                
            test_seg_label_onehot = one_hot(label, num_classes=args.classes, dim=1)
            
            metrics_values = []
            if calculate_metrics:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for func in metrics_list:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            func(y_pred=test_seg_pred_onehot,
                                y=test_seg_label_onehot)
                        results = func.aggregate()
                        if any(~torch.isfinite(results)):
                            mean_metric = results[torch.isfinite(results)].mean().item()
                        else:
                            mean_metric = results[results.nonzero()].mean().item()
                        metrics_values.append(mean_metric)
                        func.reset()
                metrics=np.array(metrics_values)*[1,0.4,0.4] #pidim 0.4 mm/px
                all_metrics.append(metrics.tolist())
            else:
                dice(test_seg_pred_onehot, test_seg_label_onehot)
                results = dice.aggregate()
                if any(~torch.isfinite(results)):
                    mean_metric = results[torch.isfinite(results)].mean().item()
                else:
                    mean_metric = results[results.nonzero()].mean().item()
                dice_results.append(mean_metric)
                dice.reset()
            
            #log visual results
            if log3d_results:
                multiclass_segmentation_np = multiclass_segmentation_pred.squeeze().cpu().numpy()
                scene_3d = log.log_3dscene_comp(multiclass_segmentation_np, label.squeeze().cpu().numpy().astype(np.int32), num_classes=32, scene_size=1024)
                font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 72, encoding="unic")
                im = Image.fromarray(scene_3d)
                dice_metric = metrics[0] if calculate_metrics else dice_results[-1]
                ImageDraw.Draw(im).text((0, 0), file_name.split('/')[-1]+'_dice_' + f'{dice_metric:.4f}', (0, 0, 0), font=font)
                if not os.path.exists(f'output/test_segmentation_results/experiments/{experiment_folder}/{configuration_name}/pyvista_preview/'):
                    os.makedirs(f'output/test_segmentation_results/experiments/{experiment_folder}/{configuration_name}/pyvista_preview/')
                im.save(f"output/test_segmentation_results/experiments/{experiment_folder}/{configuration_name}/pyvista_preview/{file_name.split('/')[-1]}.png")
            
            #save results - nifti files
            if is_save_nifti:
                if "china" in dataset:
                    save_path = f"output/public_dataset/experiments/{experiment_folder}/{configuration_name}/nifti_files"
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    pass
                else:  
                    pixdim=0.4
                    save_path = f"output/test_segmentation_results/experiments/{experiment_folder}/{configuration_name}/nifti_files"
                    file_name = file_name.split('/')[-1]
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    if crop_roi:
                        if original_shape != multiclass_segmentation_pred.shape:
                            original_image = torch.zeros_like(test_data["label"])
                            original_image[roi_slices] = multiclass_segmentation_pred
                            multiclass_segmentation_pred = original_image
                    save_nifti(multiclass_segmentation_pred, path=save_path, filename=file_name, pixdim=pixdim)
        if not calculate_metrics:    
            print(f'dice mean result: {np.array(dice_results).mean():.4f}')
        else:
            all_metrics = np.array(all_metrics)
            print('Overall results: ',' '.join([f"{m_n}-{m:.4f}±{s:.4f}," for m_n, m, s in zip(metrics_names_list, np.array(all_metrics).mean(axis=0).tolist(), np.array(all_metrics).std(axis=0).tolist())]))

print('Inference completed.')