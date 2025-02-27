import os
import yaml
from pathlib import Path
from argparse import Namespace

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

#update general configuration
update_config = experiments_config[experiment_name]['update_config'].get('general', None)
if update_config is not None:
    general_config['args'].update(update_config)

args = Namespace(**general_config['args'])

if args.comet:
    import comet_ml
    from comet_ml import Experiment
else:
    from src.commons.dummy_logger import DummyExperiment
    experiment = DummyExperiment()
    print("Comet logger false.")

import uuid
import itertools
import json
import glob
import numpy as np
import os
import random
import time
import warnings
from natsort import natsorted
from datetime import datetime

# TORCH modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import SubsetRandomSampler

#MONAI modules
from monai.networks.nets import UNet, VNet, AttentionUnet, SwinUNETR
from monai.networks.utils import one_hot
from monai.losses import DiceLoss
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric, MeanIoU, DiceMetric
from monai.metrics import CumulativeAverage
from monai.optimizers import WarmupCosineSchedule
from monai.utils import set_determinism
from monai.data import set_track_meta, ThreadDataLoader, decollate_batch
from monai.data.dataset import PersistentDataset
from monai.inferers import sliding_window_inference

from sklearn.model_selection import KFold

#my modules
from src.commons import setup_cuda
from src.commons import Logger
from src.commons import CosineAnnealingWarmupRestarts

from src.data_augmentation import Transforms
from src.data_augmentation import Transforms
from src.loss.losses import DiceCELoss, DiceFocalLoss
from src.models.resunet import ResUNet
from src.models.swin_smt.swin_smt import SwinSMT
from src.models.vsmtrans import VSmixTUnet

#REPRODUCIBLITY and precision
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
# precision for nn.modules : eg. nn.conv3d - # Nvidia Ampere 
torch.backends.cudnn.allow_tf32 = True
# precision for linear algebra - eg. interpolations and elastic transforms
torch.backends.cuda.matmul.allow_tf32 = True
if args.seed != -1:
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    set_determinism(seed=args.seed)
    # some operations cannot be made deterministic - in that case warn
    torch.use_deterministic_algorithms(mode=args.deterministic_algorithms, warn_only=True)
    if args.deterministic_algorithms:
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

#CUDA
setup_cuda(args.gpu_frac, num_threads=args.num_threads, device=args.device, visible_devices=args.visible_devices, use_cuda_with_id=args.cuda_device_id)
if args.device == 'cuda':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=int(args.cuda_device_id))
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(int(args.cuda_device_id))

#TRANSFORMS
trans = Transforms(args, device)
set_track_meta(True)

#MONAI backed dataset
#keys - ['image', "label"]

#training
datalist =[]
nifti_paths_scans = natsorted(glob.glob(os.path.join(args.data, 'scans', '**', '*.nii.gz'), recursive=True))
nifti_paths_labels = natsorted(glob.glob(os.path.join(args.data, 'labels_american', '**', '*.nii.gz'), recursive=True))
nifti_list = [{args.keys[0]: scan, args.keys[1]: label} for (scan, label) in zip(nifti_paths_scans, nifti_paths_labels)]
datalist.extend(nifti_list)
datalist = datalist[0:97]

#test
nifti_paths_test_scans = natsorted(glob.glob(os.path.join('data', 'test_data', 'scans', '*', '*.nii.gz'), recursive=False))[:20]
nifti_paths_test_labels = natsorted(glob.glob(os.path.join('data', 'test_data', 'labels', '*', '*.nii.gz'), recursive=False))[:20]
testset_datalist = [{args.keys[0]: scan, args.keys[1]: label} for (scan, label) in zip(nifti_paths_test_scans, nifti_paths_test_labels)]
    
#CACHE    
args.cache_dir = os.path.join(args.cache_dir, f"{experiment_name}_{args.classes}_{args.patch_size[0]}_{args.spatial_crop_size[0]}_{args.spatial_crop_size[1]}_{args.spatial_crop_size[2]}")
if not os.path.exists(args.cache_dir):
    os.makedirs(os.path.join(args.cache_dir, 'train'))
    os.makedirs(os.path.join(args.cache_dir, 'val'))
    os.makedirs(os.path.join(args.cache_dir, 'test'))
if args.clear_cache:
    print("Clearning cache...")
    train_cache = glob.glob(os.path.join(args.cache_dir, 'train/*.pt'))
    val_cache = glob.glob(os.path.join(args.cache_dir, 'val/*.pt'))
    test_cache = glob.glob(os.path.join(args.cache_dir, 'test/*.pt'))
    if len(train_cache) != 0:
        for file in train_cache:
            os.remove(file)
    if len(val_cache) != 0:
        for file in val_cache:
            os.remove(file)
    if len(test_cache) != 0:
        for file in test_cache:
            os.remove(file)
    print(f"Cleared cache in dir: {args.cache_dir}, train: {len(train_cache)} files, val: {len(val_cache)} files, test: {len(test_cache)} files.")
    
#Persistent cached dataset
train_dataset = PersistentDataset(datalist, trans.train_transform, cache_dir=os.path.join(args.cache_dir, 'train'))
val_dataset = PersistentDataset(datalist, trans.val_transform, cache_dir=os.path.join(args.cache_dir, 'val'))
test_dataset = PersistentDataset(testset_datalist, trans.test_transform, cache_dir=os.path.join(args.cache_dir, 'test'))

#DATA SUBSETS SPLIT
#JSON FILE based split
training_ids = []
validation_ids = []
if args.use_json_split:
    json_split_path = 'configs/data_split.json'
    with open(json_split_path) as f:
        d = json.load(f)   
    #shuffle data only when reproducibility seed is set
    if args.seed != -1:
        random.shuffle(d['training'])
        random.shuffle(d['validation'])
    train_ids = [[idx for idx, s in enumerate(nifti_paths_scans) if visit_id in s] for visit_id in d['training']]
    val_ids = [[idx for idx,s in enumerate(nifti_paths_scans) if visit_id in s] for visit_id in d['validation']]
    train_ids = list(itertools.chain(*train_ids))
    val_ids = list(itertools.chain(*val_ids))
    training_ids.append(train_ids)
    validation_ids.append(val_ids)

##########################################################################################################
# TRAINING_STEP
def training_step(batch_idx, train_data):
    
        #model output
        output_logit = model(train_data["image"])
        
        #loss - segmentation and classification
        dice, cross_entropy = criterion(output_logit, train_data['label'])
        loss = dice + cross_entropy
        loss.backward()
         
        #optimization step with gradient accumulation for accum_iter of batches
        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm, norm_type=2.0)
            optimizer.step()
            optimizer.zero_grad() 

        #model predictions
        if (epoch+1) % args.log_metrics_interval == 0 or (epoch+1) % args.log_3d_scene_interval_training == 0 or (epoch+1) % args.log_slice_interval == 0:
            seg_pred = torch.softmax(output_logit, dim=1).argmax(dim=1, keepdim=True).long()
        
        #METRICS
        #calculate metrics every nth epoch
        if (epoch+1) % args.log_metrics_interval == 0:
            binary_gt = (train_data['label']>=1).long()
            binary_pred = (seg_pred>=1).long()
            #segmentation - BINARY DATA
            for func in seg_metrics:
                 func(y_pred=binary_pred, y=binary_gt)
            #segmentation_multiclass
            if (epoch+1) >= args.multiclass_metrics_epoch and (epoch+1) % args.multiclass_metrics_interval == 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for func in seg_metrics_multiclass:
                        func(y_pred=one_hot(seg_pred, num_classes=args.classes, dim=1), y=one_hot(train_data['label'], num_classes=args.classes, dim=1))
        
            #aggregate on epoch's end
            if (batch_idx+1) == len(train_loader):
                seg_metric_results = [func.aggregate().mean().item() for func in seg_metrics]
                if (epoch+1) >= args.multiclass_metrics_epoch and (epoch+1) % args.multiclass_metrics_interval == 0:
                    seg_multiclass_metric_results = [func.aggregate().mean().item() for func in seg_metrics_multiclass]
                    
        #log running average for loss
        batch_size = train_data["image"].shape[0]
        train_loss_cum.append(loss.item(), count=batch_size)
        train_wce_loss_cum.append(cross_entropy.item(), count=batch_size)
        train_ggwd_loss_cum.append(dice.item(), count=batch_size)
        
        #log running average for metrics
        if (epoch+1) % args.log_metrics_interval == 0 and (batch_idx+1) == len(train_loader):
            train_dice_cum.append(seg_metric_results[0], count=len(train_loader))
            train_assd_cum.append(args.pixdim*seg_metric_results[1], count=len(train_loader))
            train_hd95_cum.append(args.pixdim*seg_metric_results[2], count=len(train_loader))
        
        #CONSOLE PRINT
        #loss
        if (batch_idx+1) % args.log_batch_interval == 0:
            print(f" Batch: {batch_idx + 1:02d}/{len(train_loader)}: Loss: {loss.item():.4f} - seg_dice: {dice.item():.4f}, seg_ce: {cross_entropy.item():.4f}")
        #avg loss
        if (batch_idx+1) == len(train_loader):
            print(f" Batch: {batch_idx + 1:02d}/{len(train_loader)}: Average Loss: {train_loss_cum.aggregate().mean().item():.4f}.")
            #metrics
            if (epoch+1) % args.log_metrics_interval == 0:
                print(f" _Metrics_:\n"
                      f"  * Seg.: dice: {seg_metric_results[0]:.4f}, ASSD: {args.pixdim*seg_metric_results[1]:.4f}, HD95: {args.pixdim*seg_metric_results[2]:.4f}.")
            if (epoch+1) >= args.multiclass_metrics_epoch and (epoch+1) % args.multiclass_metrics_interval == 0:
                train_dice_multiclass_cum.append(seg_multiclass_metric_results[0], count=len(train_loader))
                print(f"  * Seg. multiclass - dice: {seg_multiclass_metric_results[0]:.4f}, ASSD: {args.pixdim*seg_multiclass_metric_results[1]:.4f}, HD95: {args.pixdim*seg_multiclass_metric_results[2]:.4f}.")
            
        #COMET ML log
        if (args.is_log_image or args.is_log_3d) and (batch_idx+1) == 10:
            if (epoch+1) % args.log_slice_interval == 0 or (epoch+1) % args.log_3d_scene_interval_training == 0:
                if (epoch+1) % args.log_slice_interval == 0:
                    image = train_data["image"][0].squeeze().detach().cpu().float().numpy()
                    #segmentation
                    pred_seg_np = seg_pred[0].squeeze().detach().cpu().numpy()
                    gt_seg_np = train_data['label'][0].squeeze().long().detach().cpu().numpy()
                    #create_img_log
                    image_log_out = log.log_image(pred_seg_np, gt_seg_np, image)
                    experiment.log_image(image_log_out, name=f'img_{(epoch+1):04}_{batch_idx+1:02}')
                if (epoch+1) % args.log_3d_scene_interval_training == 0 and args.is_log_3d:
                    #multiclass segmentation 3d scene log
                    pred_seg_np = seg_pred[0].squeeze().detach().cpu().float().numpy()
                    label_seg_np = train_data['label'][0].squeeze().detach().cpu().float().numpy()
                    scene_log_out = log.log_3dscene_comp(pred_seg_np, label_seg_np, num_classes=args.classes-1, scene_size=1024)
                    experiment.log_image(scene_log_out, name=f'scene_multiclass_{(epoch+1):04}_{batch_idx+1:02}')
                    
# VALIDATION STEP
def validation_step(batch_idx, val_data):

    output_logit = sliding_window_inference(val_data["image"], roi_size=args.patch_size, sw_batch_size=8, predictor=model, overlap=0.6, sw_device=device,
                                        device=device, mode='gaussian', sigma_scale=0.125, padding_mode='constant', cval=0, progress=False)
    #multiclass_segmentation 
    val_seg_pred = torch.softmax(output_logit, dim=1).argmax(dim=1, keepdim=True).long()
    val_seg_label = one_hot(val_data["label"], num_classes=args.classes, dim=1)
    
    #METRICS
    #calculate metrics every nth epoch
    if (epoch+1) % args.log_metrics_interval == 0:
        val_binary_gt = (val_data['label']>=1).long()
        val_binary_pred = (val_seg_pred>=1).long()
        #segmentation - BINARY DATA
        for func in seg_metrics:
                func(y_pred=val_binary_pred, y=val_binary_gt)
        #segmentation_multiclass
        if (epoch+1) >= args.multiclass_metrics_epoch and (epoch+1) % args.multiclass_metrics_interval == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for func in seg_metrics_multiclass:
                    func(y_pred=one_hot(val_seg_pred, num_classes=args.classes, dim=1), y=val_seg_label)
        #aggregate on epoch's end
        if (batch_idx+1) == len(val_loader):
            seg_metric_results = [func.aggregate().mean().item() for func in seg_metrics]
            if (epoch+1) >= args.multiclass_metrics_epoch and (epoch+1) % args.multiclass_metrics_interval == 0:
                seg_multiclass_metric_results = [func.aggregate().mean().item() for func in seg_metrics_multiclass]    
        
            #log running average for metrics
            val_dice_cum.append(seg_metric_results[0])
            val_assd_cum.append(args.pixdim*seg_metric_results[1])
            val_hd95_cum.append(args.pixdim*seg_metric_results[2])
        
            print(f" Validation metrics:\n"
                f"  * Seg.: dice: {seg_metric_results[0]:.4f}, ASSD: {args.pixdim*seg_metric_results[1]:.4f}, HD95: {args.pixdim*seg_metric_results[2]:.4f}.")
            # multiclass
            if (epoch+1) >= args.multiclass_metrics_epoch and (epoch+1) % args.multiclass_metrics_interval == 0:
                seg_multiclass_metric_results = [func.aggregate().mean().item() for func in seg_metrics_multiclass]
                val_dice_multiclass_cum.append(seg_multiclass_metric_results[0])
                print(f"  * Seg. multiclass - dice: {seg_multiclass_metric_results[0]:.3f}, ASSD: {args.pixdim*seg_multiclass_metric_results[1]:.3f}, HD95: {args.pixdim*seg_multiclass_metric_results[2]:.3f}.")

    if (epoch+1) % args.log_3d_scene_interval_validation == 0 and batch_idx==1:
            image = val_data["image"][0].squeeze().float().detach().cpu().numpy()
            pred_seg_np = val_seg_pred[0].squeeze().long().detach().cpu().numpy()
            gt_seg_np = val_data['label'][0].squeeze().long().detach().cpu().numpy()
            #create_img_log
            image_log_out = log.log_image(pred_seg_np, gt_seg_np, image)
            experiment.log_image(image_log_out, name=f'val_img_{(epoch+1):04}_{batch_idx+1:02}')
            #multiclass segmentation 3d scene log
            pred_seg_np = val_seg_pred[0].squeeze().long().detach().cpu().numpy()
            label_seg_np = val_data['label'][0].squeeze().long().detach().cpu().numpy()
            scene_log_out = log.log_3dscene_comp(pred_seg_np, label_seg_np, args.classes-1, scene_size=1024)
            experiment.log_image(scene_log_out, name=f'val_scene_{(epoch+1):04}_{batch_idx+1:02}')

for exp_num in range(experiments_count):
    #init experiment
    config_name = experiment_configurations[exp_num]
    print(f"Running config: {config_name}")
    log = Logger(args.classes, args.is_log_3d)
    folds = [[training_ids[i], validation_ids[i]] for i in range(1)]

    #experiment specific configuration 
    specific_config = experiments_config[experiment_name]['update_config'].get(config_name, None)
    if specific_config is not None:
        general_config['args'].update(specific_config)
    args = Namespace(**general_config['args'])
    #new keys
    args.configuration_name = config_name
    
    if args.comet:
        #create new experiment to log
        experiment = Experiment("anon", project_name="anon", workspace="anon")
        unique_experiment_name = experiment.get_name()
        tags = args.tags.split('#')
        tags += [experiment_name, config_name, f'cuda:{args.cuda_device_id}', str(os.getpid())]
        experiment.add_tags(tags)
        experiment.log_asset(experiment_config_file)
        experiment.log_asset(general_config_file)
        experiment.log_asset('src/data_augmentation.py')
        experiment.log_parameters(vars(args))
    else:
        unique_experiment_name = uuid.uuid4().hex
        
    print("--------------------")
    print (f"\n *** Starting experiment {unique_experiment_name}: {config_name} - {exp_num+1}/{experiments_count}:\n")
    print(f"Current server time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for fold, (train_ids, val_ids) in enumerate(folds):
        print(f"FOLD {fold}")
        print("-------------------")
        if fold == 1:
            break
        
        if args.use_random_sampler:
            train_subsampler = SubsetRandomSampler(train_ids)
            val_subsampler = SubsetRandomSampler(val_ids)
        else:
            train_subsampler = train_ids
            val_subsampler = val_ids

        #DATA LOADERS
        train_loader = ThreadDataLoader(train_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size, sampler=train_subsampler)
        val_loader = ThreadDataLoader(val_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size_val, sampler=val_subsampler)
        test_loader_A = ThreadDataLoader(test_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size_val, sampler=list(range(0,11)))
        test_loader_B = ThreadDataLoader(test_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size_val, sampler=list(range(11,20)))
        
        #UNET params
        feature_maps = tuple(2**i*args.n_features for i in range(0, args.unet_depth))
        strides = list((args.unet_depth-1)*(2,))
        
        #MODEL ARCHITECTURE
        if config_name == "SwinUNETR":
            model = SwinUNETR(spatial_dims=3, in_channels=1, out_channels=args.classes, img_size=args.patch_size, feature_size=48, norm_name=args.norm, use_checkpoint=False)
        elif config_name == "SwinUNETRv2":
            model = SwinUNETR(spatial_dims=3, in_channels=1, out_channels=args.classes, img_size=args.patch_size, feature_size=48, norm_name=args.norm, use_checkpoint=False, use_v2=True)
        elif config_name == "UNet":
            model = UNet(spatial_dims=3, in_channels=1, out_channels=args.classes, channels=feature_maps, strides=strides, act=("relu", {"inplace": True}), norm=args.norm, bias=False)
        elif config_name == "AttUNet":
            model = AttentionUnet(spatial_dims=3, in_channels=1, out_channels=args.classes, channels=feature_maps, strides=strides)
        elif config_name == "VNet":
            model = VNet(spatial_dims=3, in_channels=1, out_channels=args.classes, act=("relu", {"inplace": True}), bias=False)
        elif config_name == "ResUNet18":
            model = ResUNet(spatial_dims=3, in_channels=1, out_channels=args.classes, act='relu', norm=args.norm, bias=False, backbone_name='resnet18')
        elif "ResUNet34" in config_name:
            model = ResUNet(spatial_dims=3, in_channels=1, out_channels=args.classes, act='relu', norm=args.norm, bias=False, backbone_name='resnet34')
        elif "ResUNet50" in config_name:
            model = ResUNet(spatial_dims=3, in_channels=1, out_channels=args.classes, act='relu', norm=args.norm, bias=False, backbone_name='resnet50')
        elif "SwinSMT" in config_name:
            model = SwinSMT(spatial_dims=3, in_channels=1, out_channels=args.classes, img_size=args.patch_size, feature_size=48, norm_name=args.norm, use_checkpoint=False, use_moe=True, num_experts=args.num_experts, use_v2=True)
        elif "VSmTrans" in config_name:
            model = VSmixTUnet(spatial_dims=3, in_channels=1, out_channels=args.classes, img_size=args.patch_size, feature_size=48, norm_name=args.norm, drop_rate=0, attn_drop_rate=0, drop_path_rate=0)

            
        if args.parallel:
            model = nn.DataParallel(model).to(device)
        else:
            model = model.to(device)

        # OPTIMIZER
        if args.optimizer == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
        elif args.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.adam_ams, eps=args.adam_eps)
        elif args.optimizer == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_eps)
        else:
            raise NotImplementedError(f"There are no implementation of: {args.optimizer}")
        
        if args.weighted_cls:
            weights = torch.from_numpy(np.load('src/losses/class_weights.npy')).to(dtype=torch.float32, device=device)
            weights[0]=args.background_weight
            assert(len(weights) == args.classes)
        else:
            weights=None
            
        if args.seg_loss_name=="DiceCELoss":
            criterion = DiceCELoss(to_onehot_y=True, softmax=True, ce_weight=weights, include_background=True)
        elif args.seg_loss_name=="DiceFocalLoss":
            criterion = DiceFocalLoss(to_onehot_y=True, softmax=True, gamma=args.focal_gamma, focal_weight=weights)
        else:
            raise NotImplementedError(f"There is no implementation of: {args.seg_loss_name}")

        #SCHEDULER
        if args.scheduler_name == 'annealing':
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, verbose=True, eta_min=1e-6)
        elif args.scheduler_name == 'warmup':
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, warmup_multiplier=0, t_total=args.epochs)
        elif args.scheduler_name == "warmup_restarts":
            scheduler = CosineAnnealingWarmupRestarts(optimizer, warmup_steps=args.warmup_steps, first_cycle_steps=args.first_cycle_steps, cycle_mult=(1/2), gamma=args.scheduler_gamma, max_lr=args.lr, min_lr=args.min_lr) 
        
        if args.continue_training:
            state_dict = torch.load(args.trained_model, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            scheduler.load_state_dict(state_dict['lr_scheduler'])
            args.start_epoch = state_dict['epoch']
            print(f'Loaded model, optimizer, starting with epoch: {args.start_epoch}')
        
        #METRICS
        reduction='mean_batch'
        seg_metrics = [
                DiceMetric(include_background=args.include_background_metrics, reduction=reduction),
                # MeanIoU(include_background=args.include_background_metrics, reduction=reduction),
                SurfaceDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean', reduction=reduction, symmetric=True),
                HausdorffDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean', percentile=None, get_not_nans=False, directed=False, reduction=reduction),
                # HausdorffDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean', percentile=95, get_not_nans=False, directed=True, reduction=reduction)
                ]
        seg_metrics_multiclass = [
                DiceMetric(include_background=args.include_background_metrics, reduction=reduction),
                # MeanIoU(include_background=args.include_background_metrics, reduction=reduction),
                SurfaceDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean', reduction=reduction, symmetric=True),
                HausdorffDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean', percentile=None, get_not_nans=False, directed=False, reduction=reduction),
                # HausdorffDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean', percentile=95, get_not_nans=False, directed=True, reduction=reduction)
                ]

        #RUNNING_AVERAGES
        #training loss
        train_loss_cum = CumulativeAverage()
        train_wce_loss_cum = CumulativeAverage()
        train_ggwd_loss_cum = CumulativeAverage()
        training_loss_cms = [train_loss_cum, train_wce_loss_cum, train_ggwd_loss_cum]
        #training metrics
        train_dice_cum = CumulativeAverage()
        # train_jac_cum = CumulativeAverage()
        train_assd_cum = CumulativeAverage()
        train_hd95_cum = CumulativeAverage()
        train_dice_multiclass_cum = CumulativeAverage()
        training_metrics_cms = [train_dice_cum, train_assd_cum, train_hd95_cum]
        #validation metrics
        val_dice_cum = CumulativeAverage()
        # val_jac_cum = CumulativeAverage()
        val_assd_cum = CumulativeAverage()
        val_hd95_cum = CumulativeAverage()
        val_dice_multiclass_cum = CumulativeAverage()
        val_metrics_cms = [val_dice_cum, val_assd_cum, val_hd95_cum]
        
        with experiment.train():

            best_dice_score = 0.0
            best_dice_val_score = 0.0
            accum_iter = args.gradient_accumulation
            
            for epoch in range(args.start_epoch, args.epochs):
                start_time_epoch = time.time()
                print(f"Starting epoch {epoch + 1}")
                
                #TRAINING
                model.train()
                for batch_idx, train_data in enumerate(train_loader):
                        training_step(batch_idx, train_data)
                epoch_time=time.time() - start_time_epoch

                #RESET METRICS after training
                _ = [func.reset() for func in seg_metrics]
                if (epoch+1) >= args.multiclass_metrics_epoch:
                    _ = [func.reset() for func in seg_metrics_multiclass]
                    
                #VALIDATION
                model.eval()
                with torch.no_grad():
                    if (epoch+1) % args.validation_interval == 0 and epoch != 0:
                        print("Starting validation...")
                        start_time_validation = time.time()
                        for batch_idx, val_data in enumerate(val_loader):
                            validation_step(batch_idx, val_data)
                        val_time=time.time() - start_time_validation
                        print( f"Validation time: {val_time:.2f}s")

                    #RESET METRICS after validation
                    _ = [func.reset() for func in seg_metrics]
                    if (epoch+1) >= args.multiclass_metrics_epoch:
                        _ = [func.reset() for func in seg_metrics_multiclass]
                    
                    #AGGREGATE RUNNING AVERAGES
                    train_loss_agg = [cum.aggregate() for cum in training_loss_cms]
                    if (epoch+1) % args.log_metrics_interval == 0:
                        train_metrics_agg = [cum.aggregate() for cum in training_metrics_cms]
                    if (epoch+1) % args.validation_interval == 0:
                        val_metrics_agg = [cum.aggregate() for cum in val_metrics_cms]
                    if (epoch+1) >= args.multiclass_metrics_epoch and (epoch+1) % args.multiclass_metrics_interval == 0:
                        train_dice_multiclass_agg = train_dice_multiclass_cum.aggregate()
                        train_dice_multiclass_cum.reset()
                        experiment.log_metric("train_dice_multiclass", train_dice_multiclass_agg, epoch=epoch)
                        if (epoch+1) % args.validation_interval == 0:
                            val_dice_multiclass_agg = val_dice_multiclass_cum.aggregate()
                            val_dice_multiclass_cum.reset()
                            experiment.log_metric("val_dice_multiclass", val_dice_multiclass_agg, epoch=epoch)
                    else:
                        train_dice_multiclass_agg = 0.0
                        val_dice_multiclass_agg = 0.0
                            
                    #reset running averages
                    _ = [cum.reset() for cum in training_loss_cms]
                    _ = [cum.reset() for cum in training_metrics_cms]
                    if (epoch+1) % args.validation_interval == 0:
                        _ = [cum.reset() for cum in val_metrics_cms]

                    scheduler.step()#one step per epoch

                    #LOG METRICS TO COMET
                    if args.scheduler_name == "warmup":
                        experiment.log_metric("lr_rate", scheduler.get_last_lr(), epoch=epoch)
                    elif args.scheduler_name == "warmup_restarts":
                        experiment.log_metric("lr_rate", scheduler.get_lr(), epoch=epoch)
                    experiment.log_current_epoch(epoch)
                    #loss - total and multitask elements
                    experiment.log_metric("train_loss", train_loss_agg[0], epoch=epoch)
                    experiment.log_metric("train_wce_loss", train_loss_agg[1], epoch=epoch)
                    experiment.log_metric("train_ggwd_loss", train_loss_agg[2], epoch=epoch)
                    #train metrics
                    if (epoch+1) % args.log_metrics_interval == 0:
                        experiment.log_metric("train_dice", train_metrics_agg[0], epoch=epoch)
                        experiment.log_metric("train_assd", train_metrics_agg[1], epoch=epoch)
                        experiment.log_metric("train_hd95", train_metrics_agg[2], epoch=epoch)
                    #val metrics
                    if (epoch+1) % args.validation_interval == 0:
                        experiment.log_metric("val_dice", val_metrics_agg[0], epoch=epoch)
                        experiment.log_metric("val_assd", val_metrics_agg[1], epoch=epoch)
                        experiment.log_metric("val_hd95", val_metrics_agg[2], epoch=epoch)
                        

                    # CHECKPOINTS SAVE
                    new_best_on_val=False
                    if args.save_checkpoints:
                        directory = os.path.join("checkpoints", args.checkpoint_dir, 'named_experiments', f"{unique_experiment_name}_{config_name}")
                        if not os.path.exists(directory):
                            os.makedirs(directory)

                        # save best TRAIN model
                        if (epoch+1) % args.log_metrics_interval == 0:
                            if best_dice_score < train_metrics_agg[0]:
                                save_path = f"{directory}/model-{config_name}-{args.classes}class-fold-{fold}_current_best_train.pt"
                                torch.save({
                                        'epoch': (epoch),
                                        'model_state_dict': model.state_dict(),
                                        'model_val_dice': train_metrics_agg[0],
                                        'model_val_assd': args.pixdim*train_metrics_agg[1]
                                        }, save_path)
                                best_dice_score = train_metrics_agg[0]
                                print(f"Current best train dice score {best_dice_score:.4f}. Model saved!")
                                        
                        # save best VALIDATION score
                        if (epoch+1) % args.validation_interval == 0:
                            if best_dice_val_score < val_metrics_agg[0]:
                                new_best_on_val=True
                                save_path = f"{directory}/model-{config_name}-{args.classes}class-fold-{fold}_current_best_val.pt"
                                torch.save({
                                    'epoch': (epoch),
                                    'model_state_dict': model.state_dict(),
                                    'model_val_dice': val_metrics_agg[0],
                                    'model_val_assd': args.pixdim*val_metrics_agg[1],
                                    'model_val_dice_multiclass': val_dice_multiclass_agg
                                    }, save_path)
                                best_dice_val_score = val_metrics_agg[0]
                                print(f"Current best validation dice score {best_dice_val_score:.4f}. Model saved!")

                        #save based on SAVE INTERVAL
                        if (epoch+1) % args.save_interval == 0 and epoch != 0:
                            save_path = f"{directory}/model-{config_name}-{args.classes}class-fold-{fold}_val_{val_metrics_agg[0]:.4f}_train_{train_metrics_agg[0]:.4f}_epoch_{(epoch+1):04}.pt"
                            #save based on optimiser save interval
                            if args.save_optimizer and epoch % args.save_optimiser_interval == 0 and epoch != 0:
                                torch.save({
                                    'epoch': (epoch),
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'lr_scheduler' : scheduler.state_dict(),
                                    'model_train_dice': train_metrics_agg[0],
                                    'model_train_assd': args.pixdim*train_metrics_agg[1],
                                    'model_val_dice': val_metrics_agg[0],
                                    'model_val_assd': args.pixdim*val_metrics_agg[1]
                                    }, save_path)
                                print("Saved optimizer and scheduler state dictionaries.")
                            else:
                                torch.save({
                                    'epoch': (epoch),
                                    'model_state_dict': model.state_dict(),
                                    'model_train_dice': train_metrics_agg[0],
                                    'model_train_assd': args.pixdim*train_metrics_agg[1],
                                    'model_val_dice': val_metrics_agg[0],
                                    'model_val_assd': args.pixdim*val_metrics_agg[1]
                                    }, save_path)
                            print(f"Interval model saved! - train_dice: {train_metrics_agg[0]:.4f}, val_dice: {val_metrics_agg[0]:.4f}, best_val_dice: {best_dice_val_score:.4f}.")
                #Final epoch report
                print(f"Epoch: {epoch+1} finished. Total training loss: {train_loss_agg[0]:.4f} - total epoch time: {epoch_time:.2f}s.")
                
            print(f"Experiment {exp_num} finished!")
