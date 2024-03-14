import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""

__all__ = ['ghost_net']  # 再被调用时，只能调用ghost_net这个类


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    min_value: 是out_channel的最小值，
               如果int(v + divisor / 2) // divisor * divisor的结果小于min_value的话，
               out_channel的值就是min_value
    这个函数的作用就是对in_channel: v做一个最低限度的通道变换，使得mew_v可以被divisor整除
    e.g: 30 ---> 32 生成的通道数量能被 4 整除
    """

    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    # print(int(v), "--->", int(new_v), "生成的通道数量能被", divisor, "整除")
    return new_v


# 就是一个带有截断的sigmoid
def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()

        self.gate_fn = gate_fn
        # 通道压缩， se_ratio=0.25
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 通道压缩为之前的1/4
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        # 激活函数
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        # x_se就是通道注意力
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


# 最关键的模块
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        # 这里采用了分离卷积，ratio=2保证输出特征层的通道数等于exp
        # self.primary_conv： kernel_size = 1
        # self.cheap_operation： groups=init_channels
        self.oup = oup
        init_channels = math.ceil(oup / ratio)      # 中间层的channel(是输入层的1 / 2)
        new_channels = init_channels * (ratio - 1)  # 输出层的channel

        # 1×1的卷积用来降维,进行特征压缩，跨通道的特征提取
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        # 3×3的逐层卷积进行线性映射，跨特征点的特征提取
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        # 将x1和x2沿着通道数堆叠
        out = torch.cat([x1, x2], dim=1)
        # 只返回需要的通道数
        return out[:, :self.oup, :, :]


"""
Ghost bottleneck主要由两个堆叠的Ghost模块组成。
第一个Ghost模块用作扩展层，增加了通道数。这里将输出通道数与输入通道数之比称为expansion ratio。
第二个Ghost模块减少通道数，以与shortcut路径匹配。然后，使用shortcut连接这两个Ghost模块的输入和输出。
"""


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        # se模块通道压缩系数
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride
        # Point-wise expansion
        # 增加指定通道数，可看做逆残差结构
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        # 如果是以下的参数的话， 不需要使用额外的卷积层进行通道和尺寸的变换
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs))

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2):
        """
        width: 1.0
        dropout: 0.2
        """
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        # 计算输出的channel大小
        output_channel = _make_divisible(16 * width, 4)
        # 416, 416, 3 -> 208, 208, 16
        # stem是起源的意思
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)

        # 逆残差结构，计算瓶颈结构的输入
        input_channel = output_channel
        stages = []
        block = GhostBottleneck  # 这是一个class
        for cfg in self.cfgs:
            # 每个layer就是一个stage
            layers = []
            """
            c: 控制输出层
            exp_size: 控制影藏层
            """
            for k, exp_size, c, se_ratio, s in cfg:
                # print(k, exp_size, c, se_ratio, s)
                # 得到输出层和隐藏层的channel, 这些channel都要能被4整除
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                # 根据cfg里的内容构件瓶颈结构
                layers.append(block(input_channel, hidden_channel, output_channel, k, s, se_ratio=se_ratio))

                # 更新下一个block（瓶颈结构）的in_channel
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        # 卷积+标准化+激活函数
        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # 构建分类层
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        # ------主要的网络层--------
        x = self.blocks(x)
        # ------------------------
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


def ghostnet(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k：卷积核大小，跨特征点的特征提取能力；
        # t：第一个ghost模块中通道数的大小，值一般比较大；
        # c：瓶颈结构最终输出通道数；
        # k,  t,   c,  SE,    s
        # stage1
        # 208, 208, 16 -> 208, 208, 16
        [[3,  16,  16,  0,    1]],

        # stage2
        # 208, 208, 16 -> 104, 104, 24
        [[3,  48,  24,  0,    2]],
        [[3,  72,  24,  0,    1]],

        # stage3
        # 208, 208, 24 -> 52, 52, 40                           feat1
        [[5,  72,  40,  0.25, 2]],
        [[5,  120, 40,  0.25, 1]],

        # stage4
        # 52, 52, 40 -> 26, 26, 80 -> 26, 26, 112              feat2
        [[3,  240, 80,  0,    2]],
        [[3,  200, 80,  0,    1],
         [3,  184, 80,  0,    1],
         [3,  184, 80,  0,    1],
         [3,  480, 112, 0.25, 1],
         [3,  672, 112, 0.25, 1]],

        # stage5
        # 26, 26, 112 -> 13, 13, 160                           feat3
        [[5,  672, 160, 0.25, 2]],
        [[5,  960, 160, 0,    1],
         [5,  960, 160, 0.25, 1],
         [5,  960, 160, 0,    1],
         [5,  960, 160, 0.25, 1]]
    ]
    return GhostNet(cfgs, **kwargs)


if __name__ == "__main__":

    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ghostnet().to(device)
    summary(model, input_size=(3,224,224))

