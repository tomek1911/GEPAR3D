import torch
from monai.networks.utils import one_hot
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference

def training_step(args, batch_idx, epoch, model, criterion, optimizer, scaler, train_data, train_loader, log, experiment,
                  train_loss_cum, seg_metrics, metric1_cum, metric2_cum, autocast_d_type):

    with torch.cuda.amp.autocast(enabled=args.use_scaler, dtype=autocast_d_type):    

        output_logit = model(train_data["image"])
        loss = criterion(output_logit, train_data['label'])
        
        if args.use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        #optimization step 
        if ((batch_idx + 1) % args.gradient_accumulation == 0) or (batch_idx + 1 == len(train_loader)):
            if args.use_scaler:
                if args.grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else: 
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        #model predictions
        if (epoch+1) % args.log_metrics_interval == 0 or (epoch+1) % args.log_3d_scene_interval_training == 0 or (epoch+1) % args.log_slice_interval == 0:
            seg_pred = (torch.sigmoid(output_logit.detach()) > 0.5).float()
        
        #METRICS
        #calculate metrics every nth epoch
        if (epoch+1) % args.log_metrics_interval == 0:
            #segmentation 
            for func in seg_metrics:
                func(y_pred=seg_pred, y=train_data['label'].long())
            #aggregate on epoch's end
            if (batch_idx+1) == len(train_loader):
                seg_metric_results = [func.aggregate().mean().item() for func in seg_metrics]
                    
        #log running average for loss
        batch_size = train_data["image"].shape[0]
        train_loss_cum.append(loss.item(), count=batch_size)
        
        #log running average for metrics
        if (epoch+1) % args.log_metrics_interval == 0 and (batch_idx+1) == len(train_loader):
            metric1_cum.append(seg_metric_results[0], count=len(train_loader))
            metric2_cum.append(seg_metric_results[1]*args.pixdim, count=len(train_loader))
        
        #CONSOLE PRINT
        #loss
        if (batch_idx+1) % args.log_batch_interval == 0:
            print(f" Batch: {batch_idx + 1:02d}/{len(train_loader)}: Loss: {loss.item():.4f}")
        #avg loss
        if (batch_idx+1) == len(train_loader):
            print(f" Batch: {batch_idx + 1:02d}/{len(train_loader)}: Average Loss: {train_loss_cum.aggregate().mean().item():.4f}.")
            #metrics
            if (epoch+1) % args.log_metrics_interval == 0:
                print(f" Metrics:\n"
                      f"  * Seg.: dice: {seg_metric_results[0]:.4f}, HD95: {seg_metric_results[1]*args.pixdim:.4f}.")
                
        #COMET ML log
        if (args.is_log_image or args.is_log_3d) and (batch_idx+1) == 10:
            if (epoch+1) % args.log_slice_interval == 0 or (epoch+1) % args.log_3d_scene_interval_training == 0:
                if (epoch+1) % args.log_slice_interval == 0:
                    image = train_data["image"][0].squeeze().detach().cpu().float().numpy()
                    #SEG - binary segmentation for foreground mask
                    pred_seg_np = seg_pred[0].squeeze().detach().cpu().numpy()
                    gt_seg_np = train_data['label'][0].squeeze().long().detach().cpu().numpy()
                    #create_img_log
                    image_log_out = log.log_image(pred_seg_np, gt_seg_np, image)
                    experiment.log_image(image_log_out, name=f'img_{(epoch+1):04}_{batch_idx+1:02}')
                if (epoch+1) % args.log_3d_scene_interval_training == 0 and args.is_log_3d:
                    #binary segmentation 3d scene log
                    pred_seg_np = seg_pred[0].squeeze().detach().cpu().float().numpy()
                    label_seg_np = train_data['label'][0].squeeze().detach().cpu().float().numpy()
                    scene_log_out = log.log_3dscene_comp(pred_seg_np, label_seg_np, num_classes=1, scene_size=1024)
                    experiment.log_image(scene_log_out, name=f'scene_binary_{(epoch+1):04}_{batch_idx+1:02}')
                    
# VALIDATION STEP
def validation_step(args, batch_idx, epoch, model, data_sample, test_loader, log, experiment, seg_metrics, metric1_cum, metric2_cum, autocast_d_type, trans, device):
    with torch.cuda.amp.autocast(enabled=args.use_scaler, dtype=autocast_d_type):
        output_logit = sliding_window_inference(data_sample["image"], roi_size=args.patch_size, sw_batch_size=8, predictor=model, overlap=0.6, sw_device=device,
                                            device=device, mode='gaussian', sigma_scale=0.125, padding_mode='constant', cval=0, progress=False)
    #multiclass_segmentation 
    val_seg_pred = [trans.post_pred(i) for i in decollate_batch(output_logit)]
    val_seg_label = [i.long() for i in decollate_batch(data_sample["label"])]
    
    for func in seg_metrics:
            func(y_pred=val_seg_pred, y=val_seg_label)

    if (batch_idx+1) == len(test_loader):
        seg_metric_results = [func.aggregate().mean().item() for func in seg_metrics]

        #log running average for metrics
        metric1_cum.append(seg_metric_results[0])
        metric2_cum.append(seg_metric_results[1]*args.pixdim)
        
        print(f" Validation metrics:\n"
              f"  * Seg.: dice: {seg_metric_results[0]:.4f}, HD95: {seg_metric_results[1]*args.pixdim:.4f}.")
        
    if (epoch+1) % args.log_3d_scene_interval_validation == 0 and batch_idx==1:
            image = data_sample["image"][0].squeeze().float().detach().cpu().numpy()
            pred_seg_np = val_seg_pred[0].squeeze().long().detach().cpu().numpy()
            gt_seg_np = data_sample['label'][0].squeeze().long().detach().cpu().numpy()
            #create_img_log
            image_log_out = log.log_image(pred_seg_np, gt_seg_np, image)
            experiment.log_image(image_log_out, name=f'val_img_{(epoch+1):04}_{batch_idx+1:02}')
            #multiclass segmentation 3d scene log
            pred_seg_np = val_seg_pred[0].squeeze().long().detach().cpu().numpy()
            label_seg_np = data_sample['label'][0].squeeze().long().detach().cpu().numpy()
            scene_log_out = log.log_3dscene_comp(pred_seg_np, label_seg_np, 1, scene_size=1024)
            experiment.log_image(scene_log_out, name=f'val_scene_{(epoch+1):04}_{batch_idx+1:02}')
            
# TEST STEP
def test_step(args, batch_idx, epoch, model, data_sample, log, experiment, seg_metrics, autocast_d_type, trans, device):
    
    with torch.cuda.amp.autocast(enabled=args.use_scaler, dtype=autocast_d_type):
        output_logit = sliding_window_inference(data_sample["image"], roi_size=args.patch_size, sw_batch_size=8, predictor=model, overlap=0.6, sw_device=device,
                                            device=device, mode='gaussian', sigma_scale=0.125, padding_mode='constant', cval=0, progress=False)
    #multiclass_segmentation 
    test_seg_pred = [trans.post_pred(i) for i in decollate_batch(output_logit)]
    test_seg_label = [i.long() for i in decollate_batch(data_sample["label"])]
    
    for func in seg_metrics:
            func(y_pred=test_seg_pred, y=test_seg_label)
        
    if (epoch+1) % args.log_3d_scene_interval_validation == 0 and batch_idx==1:
            image = data_sample["image"][0].squeeze().float().detach().cpu().numpy()
            pred_seg_np = test_seg_pred[0].squeeze().long().detach().cpu().numpy()
            gt_seg_np = data_sample['label'][0].squeeze().long().detach().cpu().numpy()
            #create_img_log
            image_log_out = log.log_image(pred_seg_np, gt_seg_np, image)
            experiment.log_image(image_log_out, name=f'test_img_{(epoch+1):04}_{batch_idx+1:02}')
            #multiclass segmentation 3d scene log
            pred_seg_np = test_seg_pred[0].squeeze().long().detach().cpu().numpy()
            label_seg_np = data_sample['label'][0].squeeze().long().detach().cpu().numpy()
            scene_log_out = log.log_3dscene_comp(pred_seg_np, label_seg_np, 1, scene_size=1024)
            experiment.log_image(scene_log_out, name=f'test_scene_{(epoch+1):04}_{batch_idx+1:02}')