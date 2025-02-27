import warnings
import torch
import torch.nn as nn
import torch.nn.functional as f

from typing import Optional, Sequence, Tuple, Union
from monai.networks.layers.factories import Act, Norm
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.simplelayers import SkipConnection
from monai.networks.blocks import MaxAvgPool

if __name__ == "__main__":
    from resnet import get_outplanes, resnet
else:
    from .resnet import get_outplanes, resnet


class ResUNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.RELU,
        norm: Union[Tuple, str] = Norm.BATCH,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        backbone_name: str = 'resnet18',
        dimensions: Optional[int] = None,
        inference_mode : bool = False
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
            feed_forward=True
        )

        backbone_layers = backbone.get_encoder_layers()    
        up_channels = get_outplanes(backbone_name)
        strides = [2,1,2,2]

        if len(up_channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(up_channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_channels = up_channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.encoder_layers = backbone_layers
        self.num_res_units=num_res_units
        self.inference_mode = inference_mode

        self.encoder_blocks = []
        self.decoder_blocks = []

        def _create_block(
            outc: int, up_channels: Sequence[int],  strides: Sequence[int], is_top: bool
        ) -> nn.Module:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            #Feature map sizes
            s = strides[0]
            in_down = up_channels[0]
            up_conv = up_channels[1]  
            concat = up_conv + in_down

            subblock: nn.Module

            if len(up_channels) > 2:
                # continue recursion down
                subblock = _create_block(up_conv, up_channels[1:], strides[1:], False)
            else:
                 # get encoder layer for downsampling path
                subblock = self.encoder_layers.pop()
                self.encoder_blocks.append(subblock)

            # get encoder layer for downsampling path
            down = self.encoder_layers.pop()
            # create layer in upsampling path
            up = self._get_up_layer(concat, outc, s, is_top)
            self.encoder_blocks.append(down)
            self.decoder_blocks.append(up)
            
        _create_block(self.out_channels, self.up_channels, self.strides, True)
        self.encoder_blocks.reverse()
        
        # assign to nn.Module
        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return nn.Sequential(down_path, SkipConnection(subblock, mode="cat"), up_path)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]


        if self.act == 'relu':
            self.act = (self.act, {"inplace": False})

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        #encoder forward
        features = []
        for layer in self.encoder_blocks:
            x = layer(x)
            features.append(x)
        features.reverse()
        
        #decoder forward
        for level, (layer, feature) in enumerate(zip(self.decoder_blocks, features[1:])):
                x = torch.cat([x, feature], dim=1)
                x = layer(x)
        
        return x
    

if __name__ == "__main__":
    
    import time
    import numpy as np
    backbone_name = 'resnet18'
    device = "cuda:0"
    model = ResUNet(spatial_dims=3,
                    in_channels=1,
                    out_channels=33, 
                    act='relu',
                    norm='instance',
                    bias=False,
                    backbone_name=backbone_name).to(device)
    
    a = 128
    batch_size = 4
    input = torch.rand(batch_size,1,a,a,a).to(device)
    print(f"Model input: {input.shape}, encoder name: {backbone_name}, device: {device}.\n")
    
    time_acc = []
    memory_acc = []
    print("Running benchmark...")
    model.eval()
    for i in range(16):
        start = time.time()
        output = model(input)
        torch.cuda.synchronize()
        t = (time.time()-start) * 1000
        #warmup
        if i > 8:
            time_acc.append(t)
            memory_acc.append(torch.cuda.memory_allocated(device) / 1024 ** 3)
        
    print(f"Forward pass avg. time: {np.array(time_acc).mean():.3f} ms")
    print(f" - Allocated gpu avg. memory: {np.array(memory_acc).mean():.1f} GB")