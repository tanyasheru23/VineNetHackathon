import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import math


class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // ratio),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel)
        )

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        mlp_avg = self.mlp(avg_pool.view(avg_pool.size(0), -1)).unsqueeze(2).unsqueeze(3)
        mlp_max = self.mlp(max_pool.view(max_pool.size(0), -1)).unsqueeze(2).unsqueeze(3)
        scale = torch.sigmoid(mlp_avg + mlp_max)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        scale = torch.sigmoid(self.conv(concat))
        return x * scale


class SEBlock(nn.Module):
    def __init__(self, channel, ratio=8):
        super(SEBlock, self).__init__()
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // ratio, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.squeeze(x) * x
        return scale


class CBAMBlock(nn.Module):
    def __init__(self, channel, ratio=8):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(channel, ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim_k):
        super(Attention, self).__init__()
        self.query = nn.Linear(dim_k, dim_k)
        self.key = nn.Linear(dim_k, dim_k)
        self.value = nn.Linear(dim_k, dim_k)
        self.dim_k = dim_k

    def forward(self, q, k, v, mask=None):
        # Apply linear transformations
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # Calculate the scale factor for the dot products
        scale = math.sqrt(self.dim_k)

        # Perform scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / scale
        # Apply softmax to get probabilities
        attention = torch.softmax(attention_scores, dim=-1)

        # Multiply by values
        output = torch.matmul(attention, v)

        return output

class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, dim_k):
        super(BottleNeck, self).__init__()
        #self.attention = Attention(dim_k)
        self.cbam_block=CBAMBlock(in_channels,ratio=8)
        self.se_block=SEBlock(in_channels,ratio=8)
        self.down_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dim_k = dim_k  # Ensure this is used or removed appropriately

    def forward(self, X):
        x1=self.cbam_block(X)
        x2=self.se_block(X)
        x3=x1+x2
        out1 = self.down_layer(x3)


        # Reshape for attention. Ensure dimensions are correctly aligned
        # B, C, H, W = out1.shape
        # q = out1.view(B, C, H * W).transpose(1, 2)  # Reshape to (B, H*W, C)
        # k = q  # Same shape for simplicity in this example
        # v = q  # Same shape for simplicity in this example


        # attention_output = self.attention(q, k, v, None)

        # # Reshape attention output to match input feature map dimensions
        # attention_output = attention_output.transpose(1, 2).view(B, C, H, W)

        # # Combine the attention output with the input feature map
        # out = out1 + attention_output

        return out1

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, X):
        return self.down_layer(X)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) # 1024 --> 512
        self.DoubleConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, X, skip_connection):
        X1 = self.up_conv(X)
        if(X1.shape != skip_connection.shape):
            X1 = TF.resize(X1, skip_connection.shape[2:])
        X2 = torch.cat((X1, skip_connection), dim=1)

        return self.DoubleConv(X2)

class FinalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalConv, self).__init__()
        self.finalConv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, X):
        return self.finalConv(X)

class UNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1):
        super(UNet, self).__init__()
        self.max_pool = nn.MaxPool2d(2, 2)
        self.down1 = Down(in_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        # self.bottleNeck = Down(512, 1024)
        self.bottleNeck = BottleNeck(512, 1024, 1024)


        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.finalConv = FinalConv(64, out_channels)
        

    def forward(self, X):

        ### DownSampling
        x1_skip = self.down1(X)
        x1 = self.max_pool(x1_skip)

        x2_skip = self.down2(x1)
        x2 = self.max_pool(x2_skip)

        x3_skip = self.down3(x2)
        x3 = self.max_pool(x3_skip)

        x4_skip = self.down4(x3)
        x4 = self.max_pool(x4_skip)


        ### BottleNeck Layer
        x5 = self.bottleNeck(x4)

        ### UpSampling
        x  = self.up1(x5, x4_skip)
        x  = self.up2(x , x3_skip)
        x  = self.up3(x , x2_skip)
        x  = self.up4(x , x1_skip)
        x  = self.finalConv(x)

        return x

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def test(X):
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 512
    img_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std = [1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ]
    )
    
    img = img_transform(X)
    output = UNet(img)
    output  = output.cpu().numpy()
    output = output.resize((1920, 960))
    cv2.imwrite(output, "image.png")
    

# if __name__ == "__main__":
#     test()