import sys
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
    
experiment_name = 'general_segmentation_methods'
experiment_configurations = experiments_config[experiment_name]['experiments']
experiments_count = len(experiment_configurations)

#general config
general_config_file = 'configs/general_config.yaml'
with open(general_config_file, 'r') as file:
    general_config = yaml.safe_load(file)
general_config['args']['experiment_name'] = "geometric_prior_multitask"

#update general configuration
update_config = experiments_config[experiment_name]['update_config'].get('general', None)
if update_config is not None:
    general_config['args'].update(update_config)

args = Namespace(**general_config['args'])

import glob
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
import pandas as pd
import warnings
import tqdm
import itertools

from PIL import Image, ImageDraw, ImageFont
from natsort import natsorted
from torch.utils.data import SubsetRandomSampler

from monai.networks.nets import UNet, VNet, AttentionUnet, SwinUNETR
from monai.data import Dataset, decollate_batch, ThreadDataLoader, DataLoader
from monai.data.dataset import PersistentDataset
from monai.inferers import sliding_window_inference
from monai.metrics import (HausdorffDistanceMetric, SurfaceDistanceMetric, MeanIoU, DiceMetric)
from monai.networks.utils import one_hot
from monai.transforms import Compose
from monai.transforms import (
    Compose,
    AsDiscrete,
    Activations
)

from src.models.resunet import ResUNet
from src.models.swin_smt.swin_smt import SwinSMT
from src.models.vsmtrans import VSmixTUnet
# from src.models.segmamba.model_segmamba.segmamba import SegMamba
from src.commons.log_image import Logger
from src.data_augmentation import Transforms
from inference_utils import load_coarse_binary_model, detect_roi
from src.utils import save_nifti


#CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
torch.set_num_threads(8)

#TRANSFORM
#inference with preprocessing - load labels to calculate metrics
trans = Transforms(args)
#post processing
post_transform_multiclass = Compose([Activations(softmax=True, dim=1), AsDiscrete(argmax=True, dim=1)])

#METRICS
reduction = 'mean_batch'
dice = DiceMetric(include_background=False, reduction=reduction)
jacc = MeanIoU(include_background=False, reduction=reduction)
hausdorf = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=None, get_not_nans=False, directed=False, reduction=reduction)
hausdorf95 = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=95, get_not_nans=False, directed=False, reduction=reduction)
asd = SurfaceDistanceMetric(include_background=False, distance_metric='euclidean', reduction=reduction, symmetric=True)

metrics_list = [dice, hausdorf, asd]
metrics_names_list = ['DSC', 'HD', 'ASSD']

#DATA
dataset = "test"

if dataset == "test":
    data_dir = 'data/test_data'
    centers = 'ABCD'
    nifti_paths_scans_A = natsorted(glob.glob(os.path.join(data_dir, 'scans', 'centerA', '*.nii.gz')))
    nifti_paths_labels_A = natsorted(glob.glob(os.path.join(data_dir, 'labels', 'centerA', '*.nii.gz')))
    nifti_paths_scans_B = natsorted(glob.glob(os.path.join(data_dir, 'scans', 'centerB', '*.nii.gz')))
    nifti_paths_labels_B = natsorted(glob.glob(os.path.join(data_dir, 'labels', 'centerB', '*.nii.gz')))
    nifti_paths_scans_C = natsorted(glob.glob(os.path.join(data_dir, 'scans', 'centerC', '*.nii.gz')))
    nifti_paths_labels_C = natsorted(glob.glob(os.path.join(data_dir, 'labels', 'centerC', '*.nii.gz')))
    nifti_paths_scans_D = natsorted(glob.glob(os.path.join(data_dir, 'scans', 'centerD', '*.nii.gz')))
    nifti_paths_labels_D = natsorted(glob.glob(os.path.join(data_dir, 'labels', 'centerD', '*.nii.gz')))

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

clear_cache = False
if args.clear_cache:
    test_cache = glob.glob(os.path.join(os.path.join('data', 'cached_datasets', 'testset', '*.pt')))
    if len(test_cache) != 0:
        for file in test_cache:
            os.remove(file)
    print(f"Cleared testset cache: {len(test_cache)} files.")

test_dataset = PersistentDataset(nifti_list, trans.test_transform, cache_dir=os.path.join('data', 'cached_datasets', 'testset'))
test_loader = DataLoader(test_dataset, num_workers=0, batch_size=1)
    
#missing_teeth is 0
all_teeth = torch.arange(1,33).long()     
target_names = [f'tooth_{i}' for i in range(1,33)] 
calculate_metrics = True
log3d_results = True
is_save_nifti = True
crop_roi=True

##  coarse binary roi detection model
# load config binary coarse unet
if crop_roi:
    config_file = 'configs/dentnet_roi_binary_coarse_segmentation.yaml'
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    args_binary = Namespace(**config['args'])
    model_roi_binary = load_coarse_binary_model(args_binary, device)

if dataset == "china":
    is_save_nifti = False
    
checkpoint_paths = sorted(glob.glob('checkpoints/general_segmentation_methods/*'))
checkpoint_configurations = [('_').join(file.split('/')[-1].split('_')[3:]) for file in checkpoint_paths]

for configuration_name in experiment_configurations:

    if log3d_results:
        log = Logger(num_classes=33, is_log_3d=True)
    checkpoint_id = [configuration_name == i for i in checkpoint_configurations].index(True)
    checkpoint_path = os.path.join(checkpoint_paths[checkpoint_id], f'model-{configuration_name}-33class-fold-0_current_best_val.pt')
    print('loading state from:', checkpoint_path)
    
    #UNET params
    n_features=args.n_features
    unet_depth=args.unet_depth
    feature_maps = tuple(2**i*n_features for i in range(0, unet_depth))
    strides = list((unet_depth-1)*(2,))
    all_metrics = []
    dice_results = []
    
    #MODEL INIT
    if configuration_name == "SwinUNETR":
        model = SwinUNETR(spatial_dims=3, in_channels=1, out_channels=args.classes, img_size=args.patch_size, feature_size=48, norm_name='instance', use_checkpoint=False)
    elif configuration_name == "SwinUNETRv2":
        model = SwinUNETR(spatial_dims=3, in_channels=1, out_channels=args.classes, img_size=args.patch_size, feature_size=48, norm_name='instance', use_checkpoint=False, use_v2=True)
    elif configuration_name == "UNet":
        model = UNet(spatial_dims=3, in_channels=1, out_channels=args.classes, channels=feature_maps, strides=strides, act=("relu", {"inplace": True}), norm="instance", bias=False)
    elif configuration_name == "AttUNet":
        model = AttentionUnet(spatial_dims=3, in_channels=1, out_channels=args.classes, channels=feature_maps, strides=strides)
    elif configuration_name == "VNet":
        model = VNet(spatial_dims=3, in_channels=1, out_channels=args.classes, act=("relu", {"inplace": True}), bias=False)
    elif configuration_name == "ResUNet18":
        model = ResUNet(spatial_dims=3, in_channels=1, out_channels=args.classes, act='relu', norm='instance', bias=False, backbone_name='resnet18')
    elif configuration_name == "ResUNet34":
        model = ResUNet(spatial_dims=3, in_channels=1, out_channels=args.classes, act='relu', norm='instance', bias=False, backbone_name='resnet34')
    elif configuration_name == "ResUNet50":
        model = ResUNet(spatial_dims=3, in_channels=1, out_channels=args.classes, act='relu', norm='instance', bias=False, backbone_name='resnet50')
    elif configuration_name == "SwinSMT":
        model = SwinSMT(spatial_dims=3, in_channels=1, out_channels=args.classes, img_size=args.patch_size, feature_size=48, norm_name=args.norm, use_checkpoint=False, use_moe=True, num_experts=32, use_v2=True)
    elif configuration_name == "VSmTrans":
        model = VSmixTUnet(spatial_dims=3, in_channels=1, out_channels=args.classes, img_size=args.patch_size, feature_size=48, norm_name=args.norm, drop_rate=0, attn_drop_rate=0, drop_path_rate=0)

    state_dict = torch.load(checkpoint_path, map_location=device)
    print('\n------------------------')
    print(f"Inference of config/method: {configuration_name},")
    print(f"checkpoint path: {checkpoint_path},")
    print(f"model from epoch: {state_dict['epoch']},")
    print(f"train DSC: {state_dict.get('model_train_dice',0.0)*100:.2f},")
    print(f"train HD: {state_dict.get('model_train_hd', 0.0):.2f},")
    print(f"val DSC: {state_dict.get('model_val_dice', 0.0)*100:.2f},")
    print(f"val HD: {state_dict.get('model_val_hd', 0.0):.2f},")
    print(f"val multi-class DSC: {state_dict.get('val_dice_multiclass_agg', 0.0)*100:.2f}.", flush=True)
    
    model.load_state_dict(state_dict['model_state_dict'], strict=True)
    model = model.to(device)
    model.eval()
    
    print(f"Running inference for method: {configuration_name}")
    with torch.no_grad():
        for idx, test_data in enumerate(tqdm.tqdm(test_loader, disable=True)):
            image, label = test_data["image"], test_data["label"]
            filename = test_data['image'].meta['filename_or_obj'][0]
            #binary roi detection
            original_shape = image.shape
            if crop_roi and 'centerC' in filename or 'centerD' in filename:
                image, label, roi_slices, roi_bounds = detect_roi(args_binary, data_sample=test_data, predictor=model_roi_binary, device=device)
            
            output = sliding_window_inference(image, roi_size=args.patch_size, sw_batch_size=8, predictor=model, overlap=0.6, sw_device=device,
                                        device=device, mode='gaussian', sigma_scale=0.125, padding_mode='constant', cval=0, progress=False)
            
            multiclass_segmentation = post_transform_multiclass(output)
            
            if calculate_metrics:
                metrics_values = []
                #multiclass
                pred_one_hot = one_hot(multiclass_segmentation, num_classes=33, dim=1).long()
                gt_one_hot = one_hot(label, num_classes=33, dim=1).long()

                pred_one_hot = pred_one_hot.to(device)
                gt_one_hot = gt_one_hot.to(device)

                for func in metrics_list:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        func(y_pred=pred_one_hot,
                            y=gt_one_hot)
                    results = func.aggregate()
                    #remove infinity errors for missing classes
                    if any(~torch.isfinite(results)):
                        mean_metric = results[torch.isfinite(results)].mean().item()
                    #calculate dice only for correctly clasified tooth
                    #tooth not found are taken into account in detection F1 score metric
                    else:
                        mean_metric = results[results.nonzero()].mean().item()
                    metrics_values.append(mean_metric)
                    func.reset()

                metrics=np.array(metrics_values)*[1,0.4,0.4] #pixdim 0.4 mm/px
                all_metrics.append(metrics.tolist())
            else:
                pred_one_hot = one_hot(multiclass_segmentation, num_classes=33, dim=1).long().to(device)
                gt_one_hot = one_hot(label, num_classes=33, dim=1).long().to(device)
                dice(pred_one_hot, gt_one_hot)
                results = dice.aggregate()
                if any(~torch.isfinite(results)):
                    mean_metric = results[torch.isfinite(results)].mean().item()
                else:
                    mean_metric = results[results.nonzero()].mean().item()
                dice_results.append(mean_metric)
                dice.reset()
            
            #log visual results
            if log3d_results:
                multiclass_segmentation_np = multiclass_segmentation.squeeze().cpu().numpy()
                scene_3d = log.log_3dscene_comp(multiclass_segmentation_np, label.squeeze().cpu().numpy(), num_classes=32, scene_size=1024)
                font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 72, encoding="unic")
                im = Image.fromarray(scene_3d)
                dice_metric = metrics[1] if calculate_metrics else dice_results[-1]
                ImageDraw.Draw(im).text((0, 0), filename.split('/')[-1]+'_dice_' + f'{dice_metric:.4f}', (0, 0, 0), font=font)
                if not os.path.exists(f'output/test_segmentation_results/experiments/general_methods/{configuration_name}/pyvista_preview/'):
                    os.makedirs(f'output/test_segmentation_results/experiments/general_methods/{configuration_name}/pyvista_preview/')
                im.save(f"output/test_segmentation_results/experiments/general_methods/{configuration_name}/pyvista_preview/{filename.split('/')[-1]}.png")
            if is_save_nifti:
                pixdim=0.4
                save_path = f"output/test_segmentation_results/experiments/general_methods/{configuration_name}/nifti_files"
                file_name = filename.split('/')[-1]
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if crop_roi:
                    if original_shape != multiclass_segmentation.shape:
                        original_image = torch.zeros_like(test_data["label"])
                        original_image[roi_slices] = multiclass_segmentation
                        multiclass_segmentation = original_image
                save_nifti(multiclass_segmentation, path=save_path, filename=file_name, pixdim=pixdim)
                
        if not calculate_metrics:    
            print(f'dice mean result: {np.array(dice_results).mean():.4f}')
        else:
            all_metrics = np.array(all_metrics)
            print('Overall results: ',' '.join([f"{m_n}-{m:.4f}±{s:.4f}," for m_n, m, s in zip(metrics_names_list, np.array(all_metrics).mean(axis=0).tolist(), np.array(all_metrics).std(axis=0).tolist())]))

print('Inference completed.')