import os
import yaml
from argparse import Namespace

#specific to experiment config
experiment_config_file = 'configs/experiments_config.yaml'
with open(experiment_config_file, 'r') as file:
    experiments_config = yaml.safe_load(file)

experiment_name = 'ablation_study'
experiment_configurations = experiments_config[experiment_name]['experiments']
experiments_count = len(experiment_configurations)

#general config
general_config_file = 'configs/geometric_prior_multitask.yaml'
with open(general_config_file, 'r') as file:
    general_config = yaml.safe_load(file)
general_config['args']['experiment_name'] = "geometric_prior_multitask"

#update general configuration based on specific experiment config
update_config = experiments_config[experiment_name]['update_config'].get('general', None)
if update_config is not None:
    general_config['args'].update(update_config)

args = Namespace(**general_config['args'])

os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
os.environ["MKL_NUM_THREADS"] = str(args.num_threads)

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
from torch.nn import MSELoss

#MONAI modules
from monai.networks.utils import one_hot
from monai.losses import DiceLoss
#regression
from monai.metrics import MSEMetric
#segmentation
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric, MeanIoU, DiceMetric
from monai.metrics import CumulativeAverage
from monai.optimizers import WarmupCosineSchedule
from monai.config import print_config
from monai.utils import set_determinism
from monai.data import set_track_meta, ThreadDataLoader, decollate_batch, DataLoader
from monai.data.dataset import PersistentDataset
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from sklearn.model_selection import KFold
from PIL import Image
                 
#my modules
from src.data_augmentation import Transforms
from src.losses.losses import GeometricalWassersteinDiceLoss, GeneralizedDiceLoss, DiceCELoss, DiceFocalLoss, get_tooth_dist_matrix
from src.losses.angular_loss import AngularLoss
from src.models.gepar3d import GEPAR3D 
from src.watershed_ablation import deep_watershed_with_voting
from src.commons import EarlyStopper
from src.commons import setup_cuda
from src.commons import Logger, log_angles
from src.commons import CosineAnnealingWarmupRestarts
    
#monai config
if args.print_config:
    print_config()
#REPRODUCIBLITY and precision
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
# precision for nn.modules : eg. nn.conv3d - # Nvidia Ampere 
torch.backends.cudnn.allow_tf32 = True
# precision for linear algebra - eg. interpolations and elastic transforms
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
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

# Average mixed precision settings, default to torch.float32
scaler = None
autocast_d_type = torch.float32 
if args.use_scaler:
    TORCH_DTYPES = {
    'bfloat16': torch.bfloat16,    
    'float16': torch.float16,     
    'float32': torch.float32
    }
    scaler = torch.cuda.amp.GradScaler()
    autocast_d_type=TORCH_DTYPES[args.autocast_dtype]
    if autocast_d_type == torch.bfloat16:
        os.environ["TORCH_CUDNN_V8_API_ENABLED"]="1"
        # detect gradient errors - debug cuda C code
    if autocast_d_type != torch.float32:
        torch.autograd.set_detect_anomaly(True)

#LOGGER
log = Logger(args.classes, args.is_log_3d)

#CUDA
setup_cuda(args.gpu_frac, num_threads=args.num_threads, device=args.device, visible_devices=args.visible_devices, use_cuda_with_id=args.cuda_device_id)
if args.device == 'cuda':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=int(args.cuda_device_id))
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(int(args.cuda_device_id))

#MONAI dataset files
#keys - ['image', "label", "seed", "edt", "edt_dir"]
if args.experiment_name == "geometric_prior_multitask":
    cache_dir = os.path.join(args.cache_dir, f"geometric_prior_multitask_ablation_{args.classes}_{args.patch_size[0]}_{args.spatial_crop_size[0]}_{args.spatial_crop_size[1]}_{args.spatial_crop_size[2]}")
    data_root_dir = args.data
    datalist =[]
    nifti_paths_scans = natsorted(glob.glob(os.path.join(data_root_dir, 'scans', '**', '*.nii.gz'), recursive=True))
    nifti_paths_labels = natsorted(glob.glob(os.path.join(data_root_dir, 'labels_american', '**', '*.nii.gz'), recursive=True))
    nifti_paths_seeds = natsorted(glob.glob(os.path.join(data_root_dir, 'seed_map_labels', '**', '*.nii.gz'), recursive=True))
    nifti_paths_edt = natsorted(glob.glob(os.path.join(data_root_dir, 'distance_map_labels', '**', '*.nii.gz'), recursive=True))
    nifti_paths_edt_direction = natsorted(glob.glob(os.path.join(data_root_dir, 'direction_map_labels', '**', '*.nii.gz'), recursive=True))
    nifti_list = [{args.keys[0]: scan, args.keys[1]: label, args.keys[2]: edt, args.keys[3]: edt_dir, args.keys[4]: seed} for (
                   scan, label, edt, edt_dir, seed) in zip(nifti_paths_scans, nifti_paths_labels, nifti_paths_edt, nifti_paths_edt_direction, nifti_paths_seeds)]
    
    datalist.extend(nifti_list)
    datalist = datalist[0:97]
elif args.experiment_name == "binary_coarse_segmentation":
    cache_dir = os.path.join(args.cache_dir, f"binary_coarse_segmentation{args.classes}_{args.patch_size[0]}_{args.patch_size[1]}_{args.patch_size[2]}")
    data_root_dir = args.data
    datalist =[]
    nifti_paths_scans = natsorted(glob.glob(os.path.join(data_root_dir, 'scans', '**', '*.nii.gz'), recursive=True))
    nifti_paths_labels = natsorted(glob.glob(os.path.join(data_root_dir, 'labels_american', '**', '*.nii.gz'), recursive=True))
    nifti_list = [{args.keys[0]: scan, args.keys[1]: label} for (scan, label) in zip(nifti_paths_scans, nifti_paths_labels)]
    datalist.extend(nifti_list)
    datalist = datalist[0:97]
    #training methods
    from src.coarse_binary_training import training_step, validation_step, test_step
else:
    raise NotImplementedError

if not os.path.exists(cache_dir):
    os.makedirs(os.path.join(cache_dir, 'train'))
    os.makedirs(os.path.join(cache_dir, 'val'))
    os.makedirs(os.path.join(cache_dir, 'test'))

if args.clear_cache:
    print("Clearing cache...")
    train_cache = glob.glob(os.path.join(cache_dir, 'train/*.pt'))
    val_cache = glob.glob(os.path.join(cache_dir, 'val/*.pt'))
    test_cache = glob.glob(os.path.join(cache_dir, 'test/*.pt'))
    if len(train_cache) != 0:
        for file in train_cache:
            os.remove(file)
    if len(val_cache) != 0:
        for file in val_cache:
            os.remove(file)
    if len(test_cache) != 0:
        for file in test_cache:
            os.remove(file)
    print(f"Cleared cache in dir: {cache_dir}, train: {len(train_cache)} files, val: {len(val_cache)} files, test: {len(test_cache)} files.")

#test dataset
nifti_paths_test_scans = natsorted(glob.glob(os.path.join('data', 'test_data', 'scans', '*', '*.nii.gz'), recursive=False))[:20]
nifti_paths_test_labels = natsorted(glob.glob(os.path.join('data', 'test_data', 'labels', '*', '*.nii.gz'), recursive=False))[:20]
testset_datalist = [{'image': scan, 'label': label} for (scan, label) in zip(nifti_paths_test_scans, nifti_paths_test_labels)]

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
    print(f"Setup train data - train samples: {len(train_ids)}, val_samples: {len(val_ids)}.")
    #small dataset subset for debugging
    if args.debug_subset:
        print(f"Using small dataset subset for debugging - train: {int(0.15*len(train_ids))}, val: {int(0.25*len(val_ids))} data samples.")
        train_ids = train_ids[:int(0.15*len(train_ids))]
        val_ids = train_ids[:int(0.25*len(val_ids))]
        args.log_batch_interval = 1
    training_ids.append(train_ids)
    validation_ids.append(val_ids)
else:
    if args.seed != -1:
        kfold = KFold(n_splits=args.split, shuffle=True, random_state=args.seed)
    else:
        kfold = KFold(n_splits=args.split, shuffle=False)
    for train_ids, val_ids in kfold.split(datalist):
        training_ids.append(train_ids)
        validation_ids.append(val_ids)
    
# TRAINING_STEP 
def training_step(batch_idx, train_data, args):
    with torch.cuda.amp.autocast(enabled=args.use_scaler, dtype=autocast_d_type):    
        #model outputs: seg, seg_multiclass, seeds, edt, edt_direction 
        seg, edt, angle = model(train_data["image"])
        
        #LOSS - optimization criterions
        loss = 0
        (seg_dice_loss, distance_map_loss, angular_loss) = tuple(torch.zeros(1) for i in range(3))
       
        seg_label = torch.where(train_data["label"] >= 1, 1, 0)
        edt = torch.sigmoid(edt)
        if is_edt:
            distance_map_loss = criterion_mse(edt, train_data["edt"])
            loss += distance_map_loss * args.loss_weights['edt']
        if is_dir:
            angular_loss = criterion_angular(angle, train_data["edt_dir"], seg_label)
            loss += angular_loss * args.loss_weights['dir']

        #multiclass segmentation loss, equal weights for dice + cross entropy
        multiclass_dice_loss, ce_loss = criterion_seg(seg, train_data['label'])
        # dice_loss_mlt = criterion_mlt_dice(seg, train_data['label']) 
        dice_loss = multiclass_dice_loss + ce_loss# + dice_loss_mlt

        loss += dice_loss * args.loss_weights['seg_mlt']
        
        #backprop
        if args.use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        #optimization step 
        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
            if args.use_scaler:
                if args.grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else: 
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2.0)
                optimizer.step()
                optimizer.zero_grad()
            
        #model predictions
        if is_edt: 
            reg_pred = treshold(edt.detach())
        seg_multiclass_pred = trans.post_pred_train(seg.detach()) #softmax, argmax

        #METRICS
        #calculate metrics every nth epoch
        if (epoch+1) % args.log_metrics_interval == 0:
            #distance map regression
            if is_edt:
                for func in edt_reg_metrics:
                    func(y_pred=reg_pred, y=train_data["edt"])
            #segmentation - BINARY DATA
            # multiclass to binary - when binary segmentation is not a optimization task use binarized multiclass
            if not is_mask:
                seg_pred = seg_multiclass_pred.clone()
                seg_pred = torch.where(seg_pred >= 1, 1, 0)
            for func in seg_metrics:
                func(y_pred=seg_pred, y=seg_label)
            #segmentation_multiclass
            if (epoch+1) >= args.multiclass_metrics_epoch and (epoch+1) % args.multiclass_metrics_interval == 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred = one_hot(seg_multiclass_pred, num_classes=args.classes, dim=1)
                    y = one_hot(train_data['label'], num_classes=args.classes, dim=1)
                    for func in seg_metrics_multiclass:
                        func(y_pred=pred, y=y)
                    del pred
                    del y

            #aggregate on epoch's end
            if (batch_idx+1) == len(train_loader):
                if is_edt:
                    edt_metric_results = [func.aggregate().mean().item() for func in edt_reg_metrics]
                if is_seed:
                    pass
                    # seed_metric_results = [func.aggregate().mean().item() for func in seed_reg_metrics]
                #binary
                seg_metric_results = [func.aggregate().mean().item() for func in seg_metrics]
                #multiclass
                if (epoch+1) >= args.multiclass_metrics_epoch and (epoch+1) % args.multiclass_metrics_interval == 0:
                    seg_multiclass_metric_results = [seg_metrics_multiclass[0].aggregate().mean().item()]
                    for func in seg_metrics_multiclass[1:]: #aggregate distance metrics
                        buffer = func.get_buffer()
                        means = 0
                        for buf in buffer:
                            #do not include nans (missing tooth in GT)
                            means+=buf[torch.isfinite(buf)].mean().item()
                        seg_multiclass_metric_results.append(means/buffer.shape[0])
                    
        #log running average for loss
        batch_size = train_data["image"].shape[0]
        train_loss_cum.append(loss.item(), count=batch_size)
        edt_loss_cum.append(args.loss_weights['edt']*distance_map_loss.item(), count=batch_size)
        # seed_loss_cum.append(args.loss_weights['seed']*seed_map_loss.item(), count=batch_size)
        seg_loss_cum.append(args.loss_weights['seg']*seg_dice_loss.item(), count=batch_size)
        seg_mlt_loss_cum.append(args.loss_weights['seg_mlt']*dice_loss.item(), count=batch_size)
        angle_loss_cum.append(args.loss_weights['dir']*angular_loss.item(), count=batch_size)  
        #log running average for metrics
        if (epoch+1) % args.log_metrics_interval == 0 and (batch_idx+1) == len(train_loader):
            if is_edt:
                train_mse_edt_cum.append(edt_metric_results[0], count=len(train_loader))
            else:
                edt_metric_results = [0 for i in range(len(edt_reg_metrics))]
            train_dice_cum.append(seg_metric_results[0], count=len(train_loader))
            train_assd_cum.append(args.pixdim*seg_metric_results[1], count=len(train_loader))
            train_hd_cum.append(args.pixdim*seg_metric_results[2], count=len(train_loader))
        
        #CONSOLE PRINT
        if (batch_idx+1) % args.log_batch_interval == 0:
            print(f" Batch: {batch_idx + 1:02d}/{len(train_loader):02d}: Loss: {loss.item():.4f} - seg. mult: {args.loss_weights['seg_mlt']*dice_loss.item():.4f}, ce: {args.loss_weights['seg_mlt']*ce_loss.item():.4f}, edt: {args.loss_weights['edt']*distance_map_loss.item():.4f}, dir: {args.loss_weights['dir']*angular_loss.item():.4f}, bin.seg.: {args.loss_weights['seg']*seg_dice_loss.item():.4f}")        
        
        if (batch_idx+1) == len(train_loader):
            print(f" Batch: {batch_idx + 1:02d}/{len(train_loader):02d}: Average Loss: {train_loss_cum.aggregate().mean().item():.4f} - seg_mlt: {seg_mlt_loss_cum.aggregate().mean().item():.4f}, edt: {edt_loss_cum.aggregate().mean().item():.4f}, dir: {angle_loss_cum.aggregate().mean().item():.4f}, bin.seg.: {seg_loss_cum.aggregate().mean().item():.4f}.")
            
            if (epoch+1) % args.log_metrics_interval == 0:
                print(f" _Metrics_:\n"
                      f"  * EDT:  mse: {edt_metric_results[0]:.4f};\n"
                    #   f"  * Seed: mse: {seed_metric_results[0]:.4f};\n"
                      f"  * Seg.: dice: {seg_metric_results[0]:.4f}, ASD: {args.pixdim*seg_metric_results[1]:.4f}, HD: {args.pixdim*seg_metric_results[2]:.4f}.\n")
                #multiclass
                if (epoch+1) >= args.multiclass_metrics_epoch and (epoch+1) % args.multiclass_metrics_interval == 0:
                    train_dice_multiclass_cum.append(seg_multiclass_metric_results[0], count=len(train_loader))
                    print(f"  * Seg. multiclass - dice: {seg_multiclass_metric_results[0]:.4f}, ASD: {args.pixdim*seg_multiclass_metric_results[1]:.4f}, HD: {args.pixdim*seg_multiclass_metric_results[2]:.4f}.")

        #COMET ML log
        if (batch_idx+1) == 1:
            if (epoch+1) % args.log_slice_interval == 0 and args.is_log_image:
                image = train_data["image"][0].squeeze().detach().cpu().float().numpy()
                #EDT - gradient map
                if is_edt:
                    pred_edt_np = reg_pred[0].squeeze().detach().cpu().float().numpy()
                    gt_edt_np = train_data["edt"][0].squeeze().detach().cpu().float().numpy()
                else:
                    pred_edt_np = np.zeros_like(image)
                    gt_edt_np = np.zeros_like(image)
              
                #SEED - seeds for watershed markers
                pred_seed_np = np.zeros_like(image)
                gt_seed_np = np.zeros_like(image)
                #SEG - binary segmentation for foreground mask
                pred_seg_np = np.zeros_like(image)
                gt_seg_np = np.zeros_like(image)
                #SEG - multitclass
                pred_seg_multiclass_np = seg_multiclass_pred[0].squeeze().detach().cpu().float().numpy()
                gt_seg_multiclass_np = train_data['label'][0].squeeze().detach().cpu().float().numpy()
                #create_img_log            
                image_log_out = log.log_image_multitask(pred_edt_np, gt_edt_np, pred_seed_np, gt_seed_np, pred_seg_np, gt_seg_np, image, pred_seg_multiclass_np, gt_seg_multiclass_np)
                experiment.log_image(image_log_out, name=f'img_{(epoch+1):04}_{batch_idx+1:02}')
                if not args.comet:
                    im = Image.fromarray(image_log_out)
                    im.save(f"logs/slices/{(epoch+1):04}_{batch_idx+1:02}.png")
                if is_dir:
                    if is_mask:
                        mask = train_data['binary_label']
                    else:
                        mask = torch.where(train_data['label'] >= 1, 1, 0)
                    image_angles = log_angles(angle[0].detach().cpu(), train_data["edt_dir"][0].detach().cpu(), mask[0].detach().cpu())
                    experiment.log_image(image_angles, name=f'img_{(epoch+1):04}_{batch_idx+1:02}_grad')
            if (epoch+1) % args.log_3d_scene_interval_training == 0 and args.is_log_3d:
                #binary semantic segmentation
                if is_mask:
                    pred_seg_np = seg_pred[0].squeeze().detach().cpu().float().numpy()
                    label_seg_np = train_data['binary_label'][0].squeeze().detach().cpu().float().numpy()
                    scene_log_out = log.log_3dscene_comp(pred_seg_np, label_seg_np, num_classes=1, scene_size=1024)
                    experiment.log_image(scene_log_out, name=f'scene_{(epoch+1):04}_{batch_idx+1:02}')
                #multiclass segmentation
                pred_seg_np = seg_multiclass_pred[0].squeeze().detach().cpu().float().numpy()
                label_seg_np = train_data['label'][0].squeeze().detach().cpu().float().numpy()
                scene_log_out = log.log_3dscene_comp(pred_seg_np, label_seg_np, num_classes=32, scene_size=1024)
                experiment.log_image(scene_log_out, name=f'scene_multiclass_{(epoch+1):04}_{batch_idx+1:02}')

# VALIDATION STEP
def validation_step(batch_idx, val_data, args):
    with warnings.catch_warnings():
        with torch.cuda.amp.autocast(enabled=args.use_scaler, dtype=autocast_d_type):
            seg, edt, angle = sliding_window_inference(val_data["image"], roi_size=args.patch_size, sw_batch_size=8, predictor=model, 
                                                       overlap=0.6, sw_device=device, device=device, mode='gaussian', sigma_scale=0.125,
                                                       padding_mode='constant', cval=0, progress=False)
    if is_edt:
        val_edt_pred = [treshold(torch.sigmoid(i)) for i in decollate_batch(edt)]
        val_edt_label = [i for i in decollate_batch(val_data["edt"])]
    val_seg_multiclass_pred = [trans.post_pred(i) for i in decollate_batch(seg)]

    #METRICS
    if is_edt:
        for func in edt_reg_metrics:
            func(y_pred=val_edt_pred, y=val_edt_label)
    # multiclass to binary - when binary classification is not a task use binarized multiclass to asses model generalization 
    if not is_mask:
        val_seg_pred = [torch.where(i >= 1, 1, 0) for i in val_seg_multiclass_pred]
        val_seg_label =  [torch.where(i >= 1, 1, 0) for i in val_data['label']]
    #binary metrics
    for func in seg_metrics:
        func(y_pred=val_seg_pred, y=val_seg_label)
    #multiclass metrics
    if (epoch+1) >= args.multiclass_metrics_epoch and (epoch+1) % args.multiclass_metrics_interval == 0:
        val_seg_multiclass_label = [trans.post_pred_labels(i) for i in decollate_batch(val_data["label"])]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for func in seg_metrics_multiclass:
                func(y_pred=[one_hot(i, num_classes=args.classes, dim=0) for i in val_seg_multiclass_pred], y=val_seg_multiclass_label)

    if (batch_idx+1) == len(val_loader):
        edt_metric_results = [0 for i in range(len(edt_reg_metrics))]
        if is_edt:
            edt_metric_results = [func.aggregate().mean().item() for func in edt_reg_metrics]
        seg_metric_results = [func.aggregate().mean().item() for func in seg_metrics]

        #log running average for metrics
        val_dice_cum.append(seg_metric_results[0])
        val_assd_cum.append(args.pixdim*seg_metric_results[1])
        val_hd_cum.append(args.pixdim*seg_metric_results[2])
        val_mse_edt_cum.append(edt_metric_results[0])
        # val_mse_seed_cum.append(seed_metric_results[0])
        
        print(f" Validation metrics:\n"
            f"  * EDT:  mse: {edt_metric_results[0]:.4f};\n"
            f"  * Seg.: dice: {seg_metric_results[0]:.4f}, ASD: {args.pixdim*seg_metric_results[1]:.4f}, HD: {args.pixdim*seg_metric_results[2]:.4f}.")
        # multiclass
        if (epoch+1) >= args.multiclass_metrics_epoch and (epoch+1) % args.multiclass_metrics_interval == 0:
            seg_multiclass_metric_results = [seg_metrics_multiclass[0].aggregate().mean().item()]
            for func in seg_metrics_multiclass[1:]: #aggregate distance metrics
                buffer = func.get_buffer()
                means = 0
                for buf in buffer:
                    #do not include nans (missing tooth in GT)
                    means+=buf[torch.isfinite(buf)].mean().item()
                seg_multiclass_metric_results.append(means/buffer.shape[0])
            val_dice_multiclass_cum.append(seg_multiclass_metric_results[0])
            print(f"  * Seg. multiclass - dice: {seg_multiclass_metric_results[0]:.4f}, ASD: {args.pixdim*seg_multiclass_metric_results[1]:.4f}, HD: {args.pixdim*seg_multiclass_metric_results[2]:.4f}.")

    if (epoch+1) % args.log_3d_scene_interval_validation == 0 and batch_idx==1:
            #logged is ONLY first sample from batch - idx=0
            image = val_data["image"][0].squeeze().detach().cpu().float().numpy()
            #EDT - gradient map
            if is_edt:
                pred_edt_np = val_edt_pred[0].squeeze().detach().cpu().float().numpy()
                gt_edt_np = val_data["edt"][0].squeeze().detach().cpu().float().numpy()
            else:
                pred_edt_np = np.zeros_like(image)
                gt_edt_np = np.zeros_like(image)
            #SEED - seeds for watershed markers
            #PLACEHOLDER - no seed task
            pred_seed_np = np.zeros_like(image)
            gt_seed_np = np.zeros_like(image)
            #SEG - binary segmentation for foreground mask
            #PLACEHOLDER - no binary segmentation task
            pred_seg_np = np.zeros_like(image)
            gt_seg_np = np.zeros_like(image)
            #SEG - multiclass segmentation
            pred_seg_multiclass_np = val_seg_multiclass_pred[0].squeeze().detach().cpu().float().numpy()
            gt_seg_multiclass_np = val_data['label'][0].squeeze().detach().cpu().float().numpy()
            #create_img_log
            image_log_out = log.log_image_multitask(pred_edt_np, gt_edt_np, pred_seed_np, gt_seed_np, pred_seg_np, gt_seg_np, image, pred_seg_multiclass_np, gt_seg_multiclass_np)
            if not args.comet:
                im = Image.fromarray(image_log_out)
                im.save(f"logs/slices/{(epoch+1):04}_{batch_idx+1:02}.png")
            experiment.log_image(image_log_out, name=f'val_img_{(epoch+1):04}_{batch_idx+1:02}')
            if is_dir:
                if is_mask:
                    mask = val_data['binary_label']
                else:
                    mask = torch.where(val_data["label"] >= 1, 1, 0)
                image_angles = log_angles(angle[0].detach().cpu(), val_data["edt_dir"][0].detach().cpu(), mask[0].detach().cpu())
                experiment.log_image(image_angles, name=f'val_img_{(epoch+1):04}_{batch_idx+1:02}_direction')
            #binary 3d
            #PLACEHOLDER
            # scene_log_out = log.log_3dscene_comp(pred_seg_np, gt_seg_np, 1, scene_size=1024)
            # experiment.log_image(scene_log_out, name=f'val_scene_binary_{(epoch+1):04}_{batch_idx+1:02}')
            #multiclass 3d
            scene_log_out = log.log_3dscene_comp(pred_seg_multiclass_np, gt_seg_multiclass_np, 32, scene_size=1024)
            experiment.log_image(scene_log_out, name=f'val_scene_multiclass_{(epoch+1):04}_{batch_idx+1:02}')

for exp_num in range(experiments_count):
    #init experiment
    config_name = experiment_configurations[exp_num]
    log = Logger(args.classes, args.is_log_3d)
    
    #experiment specific configuration 
    specific_config = experiments_config[experiment_name]['update_config'].get(config_name, None)
    if specific_config is not None:
        general_config['args'].update(specific_config)
    args = Namespace(**general_config['args'])
    #ablation config
    args.configuration_name = config_name
    is_edt = 'EDT' in args.configuration_name
    is_dir = 'DIR' in args.configuration_name
    #legacy placeholder - no mask, seed tasks TODO refector
    is_mask = 'MASK' in args.configuration_name
    is_seed = 'SEED' in args.configuration_name
    
    #TRANSFORMS and DATASET
    trans = Transforms(args, device)
    if is_seed:
        keys = args.keys
    elif is_mask or is_dir:
        keys = args.keys[:-1]
    elif is_edt:
        keys = args.keys[:-2]
    else:
        keys = args.keys[:-3]
         
    datalist_experiment = [{k:i[k] for k in keys} for i in datalist]
    train_dataset = PersistentDataset(datalist_experiment, trans.train_transform, cache_dir=os.path.join(cache_dir, 'train'))
    val_dataset = PersistentDataset(datalist_experiment, trans.val_transform, cache_dir=os.path.join(cache_dir, 'val'))
    test_dataset = PersistentDataset(testset_datalist, trans.test_transform, cache_dir=os.path.join(cache_dir, 'test'))

    if args.comet:
        #create new experiment to log
        experiment = Experiment("anon", project_name="anon", workspace="anon")
        unique_experiment_name = experiment.get_name()
        tags = args.tags.split('#')
        tags += [experiment_name, config_name, args.model_name, args.seg_loss_name, args.wasserstein_config, args.classification_loss, args.dice_loss_type, f'cuda:{args.cuda_device_id}', str(os.getpid())]
        if args.is_binary_dice:
            tags += [f'binary_{args.binary_dice_weight}']
        if args.background_penalty > 1:
            tags += [f'bgp_{args.background_penalty:.1f}']
            
        experiment.add_tags(tags)
        experiment.log_asset(experiment_config_file)
        experiment.log_asset(general_config_file)
        experiment.log_asset('src/data_augmentation.py')
        experiment.log_parameters(vars(args))
        experiment.log_parameter(name="cache_dir_final", value=cache_dir)
    else:
        unique_experiment_name = uuid.uuid4().hex
        args.batch_size=2
        args.validation_interval = 5
        args.log_batch_interval = 5
        args.log_metrics_interval = 5
        args.multiclass_metrics_interval = 5
        args.multiclass_metrics_epoch = 5
        args.log_slice_interval = 1
        args.log_3d_scene_interval_training = 5
        args.log_3d_scene_interval_validation = 5
        
    print("--------------------")
    print (f"\n *** Starting experiment {unique_experiment_name}: {config_name} - {exp_num+1}/{experiments_count}:\n")
    print(f"Current server time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    ### CROSS VALIDATION LOOP ###   
    print("--------------------")
    folds = [[training_ids[i], validation_ids[i]] for i in range(args.split)]
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
            
        if args.use_thread_loader:
            train_loader = ThreadDataLoader(train_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size, sampler=train_subsampler)
            val_loader = ThreadDataLoader(val_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size_val, sampler=val_subsampler)
            test_loader_A = ThreadDataLoader(test_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size_val, sampler=list(range(0,11)))
            test_loader_B = ThreadDataLoader(test_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size_val, sampler=list(range(11,20)))
        else:
            train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, sampler=train_subsampler)
            val_loader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size_val, sampler=val_subsampler)
            test_loader_A = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size_val, sampler=list(range(0,11)))
            test_loader_B = DataLoader(test_dataset, num_workers=args.num_workers,batch_size=args.batch_size_val, sampler=list(range(11,20)))
            
        #LOSSES
        if args.weighted_cls:
            weights = torch.from_numpy(np.load('scripts/ablation_study/src/losses/weights/class_weights.npy')).to(dtype=torch.float32, device=device)
            weights[0]=args.background_weight
            assert(len(weights) == args.classes)
        else:
            weights=None
        # GEOMETRICAL PRIOR - Statistical Shape Model
        if args.experiment_name == "geometric_prior_multitask":
            #multiclass segmentation loss
            if args.seg_loss_name=="WCE_GWD":
                wasserstein_distance_matrix = get_tooth_dist_matrix(device, config=args.wasserstein_config, background_penalty=args.background_penalty, 
                                                                    background_penalty_FP=args.background_penalty_FP, background_penalty_FN=args.background_penalty_FN,
                                                                    is_normalize=args.is_normalize_wsm)
                criterion_seg = GeometricalWassersteinDiceLoss(wasserstein_distance_matrix, class_weights=weights, classification_loss=args.classification_loss, dice_loss_type=args.dice_loss_type)
            elif args.seg_loss_name=="WCE_Dice":
                criterion_seg = DiceCELoss(include_background=args.include_background_loss, ce_weight=weights, to_onehot_y=True, softmax=True)
            elif args.seg_loss_name=="FocalDice":
                criterion_seg = DiceFocalLoss(include_background=args.include_background_loss, focal_weight=weights, to_onehot_y=True, softmax=True, gamma=args.wasserstein_distance_matrix)
            else:
                raise NotImplementedError(f"There is no implementation of: {args.seg_loss_name}")
            criterion_mse = MSELoss()
            criterion_dice = DiceLoss(include_background=args.include_background_loss, smooth_dr=1e-6, smooth_nr=1e-6)
            criterion_angular = AngularLoss(is_volume_weighted=args.is_volume_weighted_dir)
        elif args.experiment_name == "binary_coarse_segmentation":
            if args.seg_loss_name=="Dice":
                criterion_dice = DiceLoss(include_background=args.include_background_loss, sigmoid=True, smooth_dr=1e-6, smooth_nr=1e-6)
    
        #MODEL INIT
        if args.model_name == "GEPAR3D-ResUnet34":
            model = GEPAR3D(spatial_dims=3, in_channels=1, out_channels=args.classes, act='relu', norm=args.norm, bias=False, backbone_name='resnet34', configuration=args.configuration_name)
        else:
            raise NotImplementedError(f"There are no implementation of: {args.model_name}")

        if args.parallel:
            model = nn.DataParallel(model).to(device)
        else:
            model = model.to(device)

        # Optimizer
        if args.optimizer == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
        elif args.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.adam_ams, eps=args.adam_eps)
        elif args.optimizer == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_eps)
        else:
            raise NotImplementedError(f"There are no implementation of: {args.optimizer}")

        # SCHEDULER
        if args.scheduler_name == 'annealing':
            scheduler = CosineAnnealingLR(
                optimizer, T_max=args.epochs, verbose=True)
        elif args.scheduler_name == 'warmup':
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.epochs+100)
        elif args.scheduler_name == "warmup_restarts":
            scheduler = CosineAnnealingWarmupRestarts(optimizer, warmup_steps=args.warmup_steps, first_cycle_steps=int(
                args.epochs * args.first_cycle_steps), cycle_mult=0.5, gamma=args.scheduler_gamma, max_lr=args.lr, min_lr=args.min_lr)
            
        if args.continue_training:
            checkpoint_data = torch.load(args.trained_model, map_location=device)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            scheduler_state = checkpoint_data.get('scheduler_state_dict', None)
            if scheduler_state is None:
                scheduler_state = checkpoint_data.get('lr_scheduler', None)   
            if scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
            args.start_epoch = checkpoint_data['epoch']
            print(f'Loaded model, optimizer and scheduler - continue training from epoch: {args.start_epoch}')

        #METRICS
        reduction='mean_batch'
        seg_metrics = [
            DiceMetric(include_background=args.include_background_metrics, reduction=reduction),
            SurfaceDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean', reduction=reduction),
            HausdorffDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean',
                                    percentile=None, get_not_nans=False, directed=False, reduction=reduction),
        ]
        seg_metrics_multiclass = [
            DiceMetric(include_background=args.include_background_metrics, reduction=reduction, ignore_empty=True),
            SurfaceDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean', reduction=reduction),
            HausdorffDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean',
                                    percentile=None, get_not_nans=False, directed=False, reduction=reduction),
        ]
        seg_metrics_binary = [
            DiceMetric(include_background=args.include_background_metrics, reduction='none'),
            HausdorffDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean',
                                    get_not_nans=False, directed=False, reduction='none')
        ]
        
        edt_reg_metrics=[MSEMetric(reduction=reduction)] 
        seed_reg_metrics=[MSEMetric(reduction=reduction)]
        treshold = nn.Threshold(1e-3, 0)

        #RUNNING_AVERAGES
        #TRAIN loss
        train_loss_cum = CumulativeAverage()
        edt_loss_cum = CumulativeAverage()
        seed_loss_cum = CumulativeAverage()
        seg_loss_cum = CumulativeAverage()
        seg_mlt_loss_cum = CumulativeAverage()
        angle_loss_cum = CumulativeAverage()
        training_loss_cms = [train_loss_cum, edt_loss_cum, seed_loss_cum, seg_loss_cum, seg_mlt_loss_cum, angle_loss_cum]
        #TRAIN metrics
        #binary
        train_dice_cum = CumulativeAverage()
        train_assd_cum = CumulativeAverage()
        train_hd_cum = CumulativeAverage()
        train_mse_edt_cum = CumulativeAverage()
        #multiclass
        train_dice_multiclass_cum = CumulativeAverage()
        train_assd_multiclass_cum = CumulativeAverage()
        train_hd_multiclass_cum = CumulativeAverage()
        training_metrics_cms = [train_dice_cum, train_assd_cum, train_hd_cum, train_mse_edt_cum]
        training_metrics_mlt_cms = [train_dice_multiclass_cum, train_assd_multiclass_cum, train_hd_multiclass_cum]
        #VAL metrics
        #binary
        val_dice_cum = CumulativeAverage()
        val_assd_cum = CumulativeAverage()
        val_hd_cum = CumulativeAverage()
        train_assd_cum = CumulativeAverage()
        val_mse_edt_cum = CumulativeAverage()
        #multiclass
        val_dice_multiclass_cum = CumulativeAverage()
        val_assd_multiclass_cum = CumulativeAverage()
        val_hd_multiclass_cum = CumulativeAverage()
        val_metrics_cms = [val_dice_cum, val_assd_cum, val_hd_cum, val_mse_edt_cum]
        val_metrics_mlt_cms = [val_dice_multiclass_cum, val_assd_multiclass_cum, val_hd_multiclass_cum]
        
        with experiment.train():

            best_dice_score = 0.0
            best_dice_val_score = 0.0
            best_dice_multiclass_val_score = 0.0
            accum_iter = args.gradient_accumulation
            for epoch in range(args.start_epoch, args.epochs):
                start_time_epoch = time.time()
                print(f"Starting epoch {epoch + 1}")
                model.train()
                for batch_idx, train_data in enumerate(train_loader):
                    if args.classes > 1:
                        training_step(batch_idx, train_data, args)  
                    else:
                        #coarse binary training
                        training_step(args, batch_idx, epoch, model, criterion_dice, optimizer, scaler, train_data, train_loader, log, experiment,
                                    train_loss_cum, seg_metrics_binary, train_dice_cum, train_hd_cum, autocast_d_type)
                epoch_time=time.time() - start_time_epoch
                print(f" Train loop finished - total time: {epoch_time:.2f}s.")

                #RESET METRICS after training steps
                if args.classes > 1:
                    _ = [func.reset() for func in seg_metrics]
                    _ = [func.reset() for func in edt_reg_metrics]
                    _ = [func.reset() for func in seed_reg_metrics]
                    if (epoch+1) >= args.multiclass_metrics_epoch and (epoch+1) % args.multiclass_metrics_interval == 0:
                        _ = [func.reset() for func in seg_metrics_multiclass]
                else:
                    _ = [func.reset() for func in seg_metrics_binary]

                #VALIDATION
                model.eval()
                with torch.no_grad():
                    if (epoch+1) % args.validation_interval == 0 and epoch != 0:
                        print("Starting validation...")
                        start_time_validation = time.time()
                        for batch_idx, val_data in enumerate(val_loader):
                            if batch_idx % 5 == 0:
                                print(f" processing validation step - {batch_idx}/{len(val_loader)}")
                            if args.classes > 1:
                                validation_step(batch_idx, val_data, args)   
                            else:
                                #coarse binary validation
                                validation_step(args, batch_idx, epoch, model, val_data, val_loader, log, experiment,
                                                seg_metrics_binary, val_dice_cum, val_hd_cum, autocast_d_type, trans, device)
                        val_time=time.time() - start_time_validation
                        print( f"Validation time: {val_time:.2f}s")                  
                        #RESET METRICS after validation steps
                        if args.classes > 1:
                            _ = [func.reset() for func in seg_metrics]
                            _ = [func.reset() for func in edt_reg_metrics]
                            _ = [func.reset() for func in seed_reg_metrics]
                            if (epoch+1) >= args.multiclass_metrics_epoch and (epoch+1) % args.multiclass_metrics_interval == 0:
                                _ = [func.reset() for func in seg_metrics_multiclass]
                        else:
                            _ = [func.reset() for func in seg_metrics_binary]

                        #aggregate metrics after validation step
                        val_metrics_agg = [cum.aggregate() for cum in val_metrics_cms]
                        val_metrics_multiclass_agg = val_dice_multiclass_cum.aggregate()
                    
                    #AGGREGATE RUNNING AVERAGES
                    train_loss_agg = [cum.aggregate() for cum in training_loss_cms]
                    if (epoch+1) % args.log_metrics_interval == 0:
                        train_metrics_agg = [cum.aggregate() for cum in training_metrics_cms]
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
                    
                    scheduler.step()

                    #LOG METRICS TO COMET
                    if args.scheduler_name == "warmup":
                        experiment.log_metric("lr_rate", scheduler.get_last_lr(), epoch=epoch)
                    elif args.scheduler_name == "warmup_restarts":
                        experiment.log_metric("lr_rate", scheduler.get_lr(), epoch=epoch)
                    experiment.log_current_epoch(epoch)
                    #loss - total and multitask elements       
                    # training_loss_cms = [train_loss_cum, edt_loss_cum, seed_loss_cum, seg_loss_cum, seg_mlt_loss_cum, angle_loss_cum]
                    experiment.log_metric("train_loss", train_loss_agg[0], epoch=epoch)
                    experiment.log_metric("train_edt_loss", train_loss_agg[1], epoch=epoch)
                    # experiment.log_metric("train_seed_loss", train_loss_agg[2], epoch=epoch)
                    experiment.log_metric("train_seg_loss", train_loss_agg[3], epoch=epoch)
                    experiment.log_metric("train_gwdl_loss", train_loss_agg[4], epoch=epoch)
                    experiment.log_metric("train_angle_loss", train_loss_agg[5], epoch=epoch)
                    #train metrics
                    if (epoch+1) % args.log_metrics_interval == 0:
                        experiment.log_metric("train_dice", train_metrics_agg[0], epoch=epoch)
                        # experiment.log_metric("train_jac", train_metrics_agg[1], epoch=epoch)
                        experiment.log_metric("train_assd", train_metrics_agg[1], epoch=epoch)
                        experiment.log_metric("train_hd", train_metrics_agg[2], epoch=epoch)
                        experiment.log_metric("train_mse_edt", train_metrics_agg[3], epoch=epoch)
                        # experiment.log_metric("train_mse_seed", train_metrics_agg[4], epoch=epoch)
                    #val metrics
                    if (epoch+1) % args.validation_interval == 0:
                        #[val_dice_cum, val_hd_cum, val_mse_edt_cum]
                        experiment.log_metric("val_dice", val_metrics_agg[0], epoch=epoch)
                        # experiment.log_metric("val_jac", val_metrics_agg[1], epoch=epoch)
                        experiment.log_metric("val_assd", val_metrics_agg[1], epoch=epoch)
                        experiment.log_metric("val_hd", val_metrics_agg[2], epoch=epoch)
                        experiment.log_metric("val_mse_edt", val_metrics_agg[3], epoch=epoch)
                        # experiment.log_metric("val_mse_seed", val_metrics_agg[4], epoch=epoch)
                        
                    # CHECKPOINTS SAVE
                    if args.save_checkpoints:
                        weighted = 'W' if args.weighted_cls else ''
                        loss_name = f"{args.seg_loss_name}_{weighted}CE"
                        if args.seg_loss_name == 'GWD':
                            quad_penalties = '_' + args.wasserstein_config
                        else:
                            quad_penalties = ''
                        directory = f"checkpoints/{args.checkpoint_dir}/{unique_experiment_name}_{config_name}/classes_{str(args.classes)}_{loss_name}{quad_penalties}"
                        if not os.path.exists(directory):
                            os.makedirs(directory)

                        # save best TRAIN model
                        if (epoch+1) % args.log_metrics_interval == 0:
                            if best_dice_score < train_metrics_agg[0]:
                                save_path = f"{directory}/model-{args.model_name}-{args.classes}class-fold-{fold}_current_best_train.pt"
                                torch.save({
                                        'epoch': (epoch),
                                        'model_state_dict': model.state_dict(),
                                        'model_train_dice': train_metrics_agg[0],
                                        'model_train_hd': train_metrics_agg[2],
                                        'experiment_name': unique_experiment_name,
                                        'experiment_key': experiment.get_key()
                                        }, save_path)
                                best_dice_score = train_metrics_agg[0]
                                print(f"Current best train dice score {best_dice_score:.4f}. Model saved!")
                                        
                        # save best VALIDATION score
                        if (epoch+1) % args.validation_interval == 0:
                            if best_dice_val_score < val_metrics_agg[0]:
                                save_path = f"{directory}/model-{args.model_name}-{args.classes}class-fold-{fold}_current_best_val.pt"
                                torch.save({
                                    'epoch': (epoch),
                                    'model_state_dict': model.state_dict(),
                                    'model_val_dice': val_metrics_agg[0],
                                    'model_val_hd': val_metrics_agg[2],
                                    'model_val_dice_multiclass': val_dice_multiclass_agg,
                                    'experiment_name': unique_experiment_name,
                                    'experiment_key': experiment.get_key()
                                    }, save_path)
                                best_dice_val_score = val_metrics_agg[0]
                                print(f"Current best binary segmentation validation dice score {best_dice_val_score:.4f}. Model saved!")
                            if best_dice_multiclass_val_score < val_metrics_multiclass_agg:
                                save_path = f"{directory}/model-{args.model_name}-{args.classes}class-fold-{fold}_current_best_multiclass_val.pt"
                                torch.save({
                                    'epoch': (epoch),
                                    'model_state_dict': model.state_dict(),
                                    'model_val_dice': val_metrics_agg[0],
                                    'model_val_hd': val_metrics_agg[2],
                                    'model_val_dice_multiclass': val_dice_multiclass_agg,
                                    'experiment_name': unique_experiment_name,
                                    'experiment_key': experiment.get_key()
                                    }, save_path)
                                best_dice_multiclass_val_score = val_metrics_multiclass_agg
                                print(f"Current best multiclass segmentation validation dice score {best_dice_multiclass_val_score:.4f}. Model saved!")

                        #save based on SAVE INTERVAL
                        if (epoch+1) % args.save_interval == 0 and epoch != 0:
                            save_path = f"{directory}/model-{args.model_name}-{args.classes}class-fold-{fold}_val_{val_metrics_agg[0]:.4f}_train_{train_metrics_agg[0]:.4f}_epoch_{(epoch):04}.pt"
                            #save based on optimiser save interval
                            if args.save_optimizer and epoch % args.save_optimiser_interval == 0 and epoch != 0:
                                torch.save({
                                    'epoch': (epoch),
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'scheduler_state_dict' : scheduler.state_dict(),
                                    'model_train_dice': train_metrics_agg[0],
                                    'model_train_hd': train_metrics_agg[2],
                                    'model_val_dice': val_metrics_agg[0],
                                    'model_val_hd': val_metrics_agg[2],
                                    'experiment_name': unique_experiment_name,
                                    'experiment_key': experiment.get_key()
                                    }, save_path)
                                print("Saved optimizer and scheduler state dictionaries.")
                            else:
                                torch.save({
                                    'epoch': (epoch),
                                    'model_state_dict': model.state_dict(),
                                    'model_train_dice': train_metrics_agg[0],
                                    'model_train_hd': train_metrics_agg[2],
                                    'model_val_dice': val_metrics_agg[0],
                                    'model_val_hd': val_metrics_agg[2],
                                    'experiment_name': unique_experiment_name,
                                    'experiment_key': experiment.get_key()
                                    }, save_path)
                            print(f"Interval model saved! - train_dice: {train_metrics_agg[0]:.4f}, val_dice: {val_metrics_agg[0]:.4f}, best_val_dice: {best_dice_val_score:.4f}.")
                #Final epoch report
                epoch_time=time.time() - start_time_epoch
                print(f"Epoch: {epoch+1} finished. Total training loss: {train_loss_agg[0]:.4f} - total epoch time: {epoch_time:.2f}s.")
            print(f"Training finished!")
    print(f"Experiments finished! logging to comet server...")
    #wait to move logs to comet
    experiment.flush()
    experiment.end()
    print("---------------------------------------------------------\n")
    print (f"Experiment {unique_experiment_name}-{config_name} sent to server.")
    print("---------------------------------------------------------\n")