import warnings
import torch
import torch.nn as nn
import torch.nn.functional as f

from typing import Tuple, Union
from monai.networks.layers.factories import Act, Norm

if __name__ == "__main__":
    from resnet import get_outplanes, resnet
else:
    from .resnet import get_outplanes, resnet

class GEPAR3D(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: Union[Tuple, str] = Act.RELU,
        norm: Union[Tuple, str] = Norm.BATCH,
        bias: bool = True,
        backbone_name: str = 'resnet34',
        inference_mode : bool = False,
        configuration: str = "EDT_DIR"
    ) -> None:
        super().__init__()

        backbone = resnet(
            norm=norm,
            act=act,
            resnet_type=backbone_name,
            spatial_dims=spatial_dims,
            n_input_channels=in_channels,
            num_classes=out_channels,
            conv1_t_size=3,
            conv1_t_stride=2,
            no_max_pool=True,
            feed_forward=False
        )

        self.backbone_layers = nn.ModuleList(backbone.get_encoder_layers())
        up_channels = get_outplanes(backbone_name)

        self.inference_mode = inference_mode
        self.validation_mode = False
        
        #decoder configuration
        self.is_edt = 'EDT' in configuration
        self.is_dir = 'DIR' in configuration
        
        #DECODER
        ch4, ch3, ch2 = 256, 128, 64 #skip_channels: 512, 256, 128, 64, 64 (resnet34)
        conv4 = nn.Sequential(nn.ConvTranspose3d(up_channels[4]+up_channels[3], ch4, kernel_size=3, padding=1, bias=bias, stride=2, dilation=1, output_padding=1),
                              nn.InstanceNorm3d(ch4),
                              nn.ReLU(inplace=True))
        conv3 = nn.Sequential(nn.ConvTranspose3d(ch4+up_channels[2], ch3, kernel_size=3, padding=1, bias=bias, stride=2, dilation=1, output_padding=1),
                              nn.InstanceNorm3d(ch3),
                              nn.ReLU(inplace=True))
        conv2 = nn.Sequential(nn.ConvTranspose3d(ch3+up_channels[1], ch2, kernel_size=3, padding=1, bias=bias, stride=1),
                              nn.InstanceNorm3d(ch2),
                              nn.ReLU(inplace=True))
        
        self.decoder_blocks = nn.ModuleList([conv4, conv3, conv2])
        
        #instance map
        self.edt_decoder = nn.ConvTranspose3d(ch2+up_channels[0], 1, kernel_size=3, padding=1, bias=bias, stride=2, dilation=1, output_padding=1)
        self.direction_decoder = nn.ConvTranspose3d(ch2+up_channels[0], 3, kernel_size=3, padding=1, bias=bias, stride=2, dilation=1, output_padding=1)

        #segmentation
        self.segmentation_decoder = nn.Sequential(nn.ConvTranspose3d(ch3+up_channels[1], ch2, kernel_size=3, padding=1, bias=bias, stride=1),
                                                  nn.InstanceNorm3d(ch2),
                                                  nn.ReLU(inplace=True))
        self.multiclass_decoder = nn.ConvTranspose3d(ch2+up_channels[0], out_channels, kernel_size=3, padding=1, bias=bias, stride=2, dilation=1, output_padding=1)

        self._zero_cache = {}
    
    def _get_cached_zeros(self, shape: Tuple, device: torch.device) -> torch.Tensor:
        key = (shape, str(device))
        if key not in self._zero_cache:
            self._zero_cache[key] = torch.zeros(shape, device=device)
        return self._zero_cache[key]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seg_multiclass, edt, edt_direction = (None,)*3
        B, _, *s_dim = x.shape

        #ENCODER forward
        features = []
        for layer in self.backbone_layers:
            x = layer(x)
            features.append(x)
        features.reverse()
        
        #DECODER forward
        for level, (layer, feature) in enumerate(zip(self.decoder_blocks, features[1:])):
            if level == 2:
                seg = self.segmentation_decoder(torch.cat([x, feature], dim=1))
            #shared decoder 
            x = torch.cat([x, feature], dim=1)
            x = layer(x)
        
        seg_multiclass = self.multiclass_decoder(torch.cat([seg, features[-1]], dim=1))
        if self.is_edt:
            edt = self.edt_decoder(torch.cat([x, features[-1]], dim=1))   
            if self.is_dir:
                edt_direction = self.direction_decoder(torch.cat([x, features[-1]], dim=1))
                edt_direction = f.normalize(edt_direction, p=2.0, dim=1, eps=1e-12)
            else:
                edt_direction = self._get_cached_zeros((B, 3, *s_dim), x.device)
        else:
            edt = self._get_cached_zeros((B, 1, *s_dim), x.device)
            edt_direction = self._get_cached_zeros((B, 3, *s_dim), x.device)
          
        if self.inference_mode:
            return torch.softmax(seg_multiclass, dim=1), torch.sigmoid(edt)
        else:
            return seg_multiclass, edt, edt_direction
        
if __name__ == "__main__":
    
    import time
    import numpy as np
    backbone_name = 'resnet34'
    device = "cuda:1"
    model = GEPAR3D(spatial_dims=3,
                    in_channels=1,
                    out_channels=32, 
                    act='relu',
                    norm='instance',
                    bias=False,
                    backbone_name=backbone_name,
                    configuration='EDT_DIR'
                    ).to(device)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable paramters: {pytorch_trainable_params/10**6:.2f}M, all parameters: {pytorch_total_params/10**6:.2f}M.")
    
    a = 128
    batch_size = 2
    input = torch.rand(batch_size,1,a,a,a).to(device)
    print(f"Model input: {input.shape}, encoder name: {backbone_name}, device: {device}.\n")
    
    time_acc = []
    memory_acc = []
    print("Running benchmark...")
    # model.train()
    model.eval()
    with torch.no_grad():
        for i in range(64):
            start = time.time()
            output = model(input)
            # (1-output[0]).mean().backward()
            t = (time.time()-start) * 1000
            torch.cuda.synchronize()
            if i > 32:
                time_acc.append(t)
                memory_acc.append(torch.cuda.memory_allocated(device) / 1024 ** 3)
            
        print(f"Forward pass avg. time: {np.array(time_acc).mean():.2f} ms")
        print(f" - Allocated gpu avg. memory: {np.array(memory_acc).mean():.2f} GB")