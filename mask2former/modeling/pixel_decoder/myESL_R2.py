import torch
import torch.nn as nn
import torch.nn.functional as F

# The definition of the Edge-Enhanced Semantic Learning Module (EESLM).

class MYESL_R2(nn.Module):
    def __init__(self, in_channels, dj = True):
        super(MYESL_R2, self).__init__()

        self.dj = dj

        self.in_channels = in_channels

        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.ReLu_s = nn.ReLU()

        self.ReLu_l = nn.ReLU()

        # Scharr and Laplacian
        self.register_buffer('scharr_x', self.get_scharr_kernel('x'))
        self.register_buffer('scharr_y', self.get_scharr_kernel('y'))
        self.register_buffer('laplacian', self.get_laplacian_kernel())

        # 1x1 conv result
        self.fusion = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)

    def get_scharr_kernel(self, direction='x'):
        if direction == 'x':
            kernel = torch.tensor([[[-3., 0., 3.],
                                    [-10., 0., 10.],
                                    [-3., 0., 3.]]])
        else:
            kernel = torch.tensor([[[-3., -10., -3.],
                                    [0., 0., 0.],
                                    [3., 10., 3.]]])
        kernel = kernel.repeat(self.in_channels, 1, 1, 1)
        return kernel

    def get_laplacian_kernel(self):
        kernel = torch.tensor([[[0., -1., 0.],
                                [-1., 4., -1.],
                                [0., -1., 0.]]])
        kernel = kernel.repeat(self.in_channels, 1, 1, 1)
        return kernel

    def forward(self, x):

        # Scharr
        edge_x = F.conv2d(x, self.scharr_x, padding=1, groups=self.in_channels)
        edge_y = F.conv2d(x, self.scharr_y, padding=1, groups=self.in_channels)
        scharr_feat = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

        scharr_feat = self.ReLu_s(scharr_feat)

        # Laplacian
        laplacian_feat = F.conv2d(x, self.laplacian, padding=1, groups=self.in_channels)

        laplacian_feat = self.ReLu_l(laplacian_feat)

        # 3×3 Learnable Conv
        conv_feat = self.conv3x3(x)

        # Concatenate: [B, 3*C, H, W]
        fusion_input = torch.cat([scharr_feat, laplacian_feat, conv_feat], dim=1)

        # Final 1x1 conv to get back to [B, C, H, W]
        out = self.fusion(fusion_input)

        if self.dj == True:
            return out * 0.5 + x * 0.5
        elif self.dj == False:
            return out