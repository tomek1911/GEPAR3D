import torch
import torch.nn as nn
import cv2
import numpy as np
from raster_geometry import sphere, circle

class SobelFilter(nn.Module):
    def __init__(self, spatial_size : int = 3):
        super(SobelFilter, self).__init__()
        self.spatial_size = spatial_size
        if spatial_size == 2:
            self.conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv.weight = nn.Parameter(self.get_sobel_kernel(), requires_grad=False)
        elif spatial_size ==3:
            self.conv = nn.Conv3d(1, 3, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv.weight = nn.Parameter(self.get_sobel_kernel(), requires_grad=False)

    def get_sobel_kernel(self):
        assert self.spatial_size in [2, 3], "spatial_size must be 2 or 3"

        derivative_kernel_2d = torch.tensor([[-1, 0, 1]])
        smoothing_kernel_2d = torch.tensor([[1, 2, 1]])
        x_sobel_kernel_2d = derivative_kernel_2d * smoothing_kernel_2d.transpose(0,1)
        y_sobel_kernel_2d = smoothing_kernel_2d * derivative_kernel_2d.transpose(0,1)

        if self.spatial_size == 2:
            kernel_2d = torch.stack([x_sobel_kernel_2d, y_sobel_kernel_2d]).unsqueeze(1).float()
            return kernel_2d 
        elif self.spatial_size == 3:
            smoothing_kernel_3d = smoothing_kernel_2d[None]
            derivative_kernel_3d = derivative_kernel_2d[None]
            x_sobel_kernel_3d = x_sobel_kernel_2d[None] * smoothing_kernel_3d.transpose(0,2)
            y_sobel_kernel_3d = y_sobel_kernel_2d[None] * smoothing_kernel_3d.transpose(0,2)
            z_sobel_kernel_3d = smoothing_kernel_2d * smoothing_kernel_2d.transpose(0,1) * derivative_kernel_3d.transpose(0,2)
            kernel_3d = torch.stack([x_sobel_kernel_3d, y_sobel_kernel_3d, z_sobel_kernel_3d]).unsqueeze(1).float()
            return kernel_3d
        
    def forward(self, x):
        x = self.conv(x)
        if self.spatial_size == 2:
            x=x/8
        elif self.spatial_size == 3:
            x=x/16
        return x

class Sobel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        Gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x

if __name__ == "__main__":

    #2d
    sobel_2d=Sobel2()
    c = circle((10,10),4)
    c = torch.from_numpy(c).to(torch.float32).reshape((1,) + c.shape)
    b = sobel_2d(c)
    #3d  
    sobel = SobelFilter(3)
    c = sphere(shape=(10,10,10), radius=4, smoothing=False)
    c = torch.from_numpy(c).to(torch.float32).reshape((1,) + c.shape)
    b = sobel(c)
    print(b)