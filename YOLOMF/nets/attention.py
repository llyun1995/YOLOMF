import torch
import torch.nn as nn
import math
from torch.nn import Module, Conv2d, Parameter, Softmax
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

# se_block------------------------------------------------------------------------------------------------
class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# eca_block----------------------------------------------------------------------------------------------
class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# CA_block----------------------------------------------------------------------------------------------
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CA_block(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=32):
        super(CA_block, self).__init__()
        # c×1×W
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 在特征图的高和宽上进行全局平均池化，自适应全局平均池化输出的高和宽均为1。
        # c×H×1
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 在特征图的高和宽上进行全局平均池化，自适应全局平均池化输出的高和宽均为1。
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv_h = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)   # c×H×1
        self.conv_w = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)   # c×1×W

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        # n×c×H×1
        x_h = self.pool_h(x)
        # n×c×W×1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        # n×c×(W+H)×1
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        # n×c×1×W
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv_h(x_h))
        out_w = torch.sigmoid(self.conv_w(x_w))
        return short * out_w * out_h


# cbam_block-------------------------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x*self.channelattention(x)
        x = x*self.spatialattention(x)
        return x


# DANet_block------------------------------------------------------------------------------------------
__all__ = ['PAM_Module', 'CAM_Module']


class PAM_Module(Module):  # Position attention module
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        # 先经过3个卷积层生成3个新特征图B C D （尺寸不变）
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv   = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # α尺度系数初始化为0，并逐渐地学习分配到更大的权重
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)  # 对每一行进行softmax

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × (H×W) × (H×W)
        """
        m_batchsize, C, height, width = x.size()
        # B -> (N,C,HW) -> (N,HW,C)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        # C -> (N,C,HW)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        # BC，空间注意图 -> (N,HW,HW)
        energy = torch.bmm(proj_query, proj_key)
        # S = softmax(BC) -> (N,HW,HW)
        attention = self.softmax(energy)
        # D -> (N,C,HW)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        # DS -> (N,C,HW)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # torch.bmm表示批次矩阵乘法
        # output -> (N,C,H,W)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(Module):    # Channel attention module
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        # β尺度系数初始化为0，并逐渐地学习分配到更大的权重
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)  # 对每一行进行softmax

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × C × C
        """
        m_batchsize, C, height, width = x.size()
        # A -> (N,C,HW)
        proj_query = x.view(m_batchsize, C, -1)
        # A -> (N,HW,C)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        # 矩阵乘积，通道注意图：X -> (N,C,C)
        energy = torch.bmm(proj_query, proj_key)
        # 这里实现了softmax，用最后一维的最大值减去了原始数据，获得了一个不是太大的值
        # 沿着最后一维的C选择最大值，keepdim保证输出和输入形状一致，除了指定的dim维度大小为1
        # expand_as表示以复制的形式扩展到energy的尺寸
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy

        attention = self.softmax(energy_new)
        # A -> (N,C,HW)
        proj_value = x.view(m_batchsize, C, -1)
        # XA -> （N,C,HW）
        out = torch.bmm(attention, proj_value)
        # output -> (N,C,H,W)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class DANet_block(nn.Module):
    def __init__(self, in_channels):
        super(DANet_block, self).__init__()

        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_channels),
                                    nn.ReLU())
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(in_channels)  # 空间注意力模块
        self.sc = CAM_Module(in_channels)  # 通道注意力模块

        self.conv51 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_channels),
                                    nn.ReLU())


    def forward(self, x):
        # 经过一个1×1卷积降维后，再送入空间注意力模块
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        # 经过一个1×1卷积降维后，再送入通道注意力模块
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        output = sa_conv + sc_conv  # 两个注意力模块结果相加

        return output  # 输出模块融合后的结果

 