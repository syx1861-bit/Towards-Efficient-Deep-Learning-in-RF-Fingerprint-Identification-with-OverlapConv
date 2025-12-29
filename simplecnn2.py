import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentedSoftGroupConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, overlaprate_parameter, num_groups=2,
                 init_overlap_rate=0.5, alpha=5.0):
        super().__init__()
        self.C = in_channels
        self.O = out_channels
        self.Sc = num_groups
        self.alpha = alpha
        self.kernel_size = kernel_size

        self.base_width = self.C / self.Sc  # 每组基础通道数
        # self.overlap_rate = nn.Parameter(torch.tensor(init_overlap_rate))  # 可学习 overlap rate
        self.overlap_rate = overlaprate_parameter

        # 每组独立卷积，注意 in_channels 可能不同（稍后硬分割版可指定 in_chs）
        out_ch_per_group = self.O // self.Sc
        self.group_convs = nn.ModuleList([
            nn.Conv2d(in_channels=self.C, out_channels=out_ch_per_group,
                      kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
            for _ in range(self.Sc)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        alpha = torch.exp(torch.tensor(self.alpha, device=device))

        # overlap = torch.clamp(self.overlap_rate, 0.0, 2.0)
        overlap = 1.7 * torch.sigmoid(self.overlap_rate)
        group_width = self.base_width + overlap * self.C  # 实际组宽度（通道数量），非归一化
        L = group_width / C  # 归一化窗口宽度

        # 中心（归一化），与硬分组保持一致
        centers = torch.tensor([(i + 0.5) * self.base_width / C for i in range(self.Sc)], device=device)  # [Sc]
        channel_pos = torch.linspace(0.5, C - 0.5, C, device=device) / C  # [C]
        # print("channel_pos=",channel_pos)

        # 构造 softmask: [C, Sc]
        pos = channel_pos.view(-1, 1)  # [C, 1]
        centers = centers.view(1, -1)  # [1, Sc]

        left = torch.sigmoid(alpha * (pos - centers + L / 2))
        right = 1 - torch.sigmoid(alpha * (pos - centers - L / 2))
        masks = left * right  # [C, Sc]

        masks = masks.permute(1, 0).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # [Sc, 1, C, 1, 1]

        outputs = []
        for i in range(self.Sc):
            x_weighted = x * masks[i]  # [B, C, H, W]
            y = self.group_convs[i](x_weighted)
            outputs.append(y)

        return torch.cat(outputs, dim=1)  # [B, O, H, W]


class OverlapConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, num_groups=2, overlap_rate=0.5):
        super().__init__()
        self.C = in_channels
        self.O = out_channels
        self.Sc = num_groups
        self.overlap_rate = overlap_rate
        self.kernel_size = kernel_size

        # 每组基础宽度（不重叠时）
        self.base_width = self.C / self.Sc  # float
        self.group_width = int(round(self.base_width + self.overlap_rate * self.C))  # 实际每组处理通道数

        # 每组卷积层：注意 in_channels 是 group-specific！
        out_ch_per_group = self.O // self.Sc
        self.convs = nn.ModuleList()

        self.group_channels = []  # 存储每组输入通道范围（start, end）

        for i in range(self.Sc):
            center = (i + 0.5) * self.base_width
            start = int(round(center - self.group_width / 2))
            end = int(round(center + self.group_width / 2))

            start = max(0, start)
            end = min(self.C, end)
            in_ch = end - start
            self.group_channels.append((start, end))

            conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch_per_group,
                             kernel_size=self.kernel_size, stride=stride, padding=self.kernel_size // 2)
            self.convs.append(conv)
            print(F"L FOR GROUP{i}=", end - start)

    def forward(self, x):
        outputs = []
        for i in range(self.Sc):
            start, end = self.group_channels[i]
            x_i = x[:, start:end, :, :]  # 硬切片
            y_i = self.convs[i](x_i)
            outputs.append(y_i)
        return torch.cat(outputs, dim=1)  # [B, O, H, W]





class SimpleCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1,stride=4),  # -> (16, 102, 45)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # -> (16, 51, 22)

            nn.Conv2d(16, 32, kernel_size=3, padding=1,stride=2),  # -> (32, 51, 22)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # -> (32, 25, 11)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> (64, 25, 11)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # -> (64, 1, 1)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

class GhostBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GhostBlock, self).__init__()
        hidden = out_channels // 2
        self.primary = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )
        self.ghost = nn.Sequential(
            nn.Conv2d(hidden, out_channels - hidden, kernel_size=3, stride=1, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(out_channels - hidden),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary(x)
        x2 = self.ghost(x1)
        return torch.cat([x1, x2], dim=1)

class GhostBlock_soft(nn.Module):
    def __init__(self, in_channels, out_channels,overlaprate):
        super(GhostBlock_soft, self).__init__()
        hidden = out_channels // 2
        self.overlaprate = overlaprate
        self.primary = nn.Sequential(
            #nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            SegmentedSoftGroupConv(in_channels,hidden,1,1,num_groups=4,overlaprate_parameter=self.overlaprate),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )
        self.ghost = nn.Sequential(
            nn.Conv2d(hidden, out_channels - hidden, kernel_size=3, stride=1, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(out_channels - hidden),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary(x)
        x2 = self.ghost(x1)
        return torch.cat([x1, x2], dim=1)

class GhostBlock_hard(nn.Module):
    def __init__(self, in_channels, out_channels,overlaprate):
        super(GhostBlock_hard, self).__init__()
        hidden = out_channels // 2
        self.overlaprate = overlaprate
        self.primary = nn.Sequential(
            #nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            OverlapConv2d(in_channels,hidden,1,1,num_groups=4,overlap_rate=overlaprate),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )
        self.ghost = nn.Sequential(
            nn.Conv2d(hidden, out_channels - hidden, kernel_size=3, stride=1, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(out_channels - hidden),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary(x)
        x2 = self.ghost(x1)
        return torch.cat([x1, x2], dim=1)

class GhostCNN_spec(nn.Module):
    def __init__(self, num_classes=8):
        super(GhostCNN_spec, self).__init__()

        self.first_conv = nn.Conv2d(1, 64, 3, 1)
        #self.first_bn = nn.BatchNorm2d(16)
        self.first_relu = nn.ReLU6(inplace=False)
        self.features = nn.Sequential(
            GhostBlock(64, 64),        # [B, 16, 102, 45]
            nn.MaxPool2d(2),          # [B, 16, 51, 22]
            GhostBlock(64, 64),
            nn.MaxPool2d(2),  # [B, 32, 25, 11]
            GhostBlock(64, 64),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.first_relu(self.first_conv(x))
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))

class GhostCNN_spec_soft(nn.Module):
    def __init__(self, num_classes=8):
        super(GhostCNN_spec_soft, self).__init__()
        self.overlaprate = nn.Parameter(torch.tensor(-0.2))
        #self.overlaprate = nn.Parameter(torch.tensor(-0.2))
        self.first_conv = nn.Conv2d(1, 64, 3, 1)
        #self.first_bn = nn.BatchNorm2d(64)
        self.first_relu = nn.ReLU6(inplace=False)
        self.features = nn.Sequential(
            GhostBlock_soft(64, 64,overlaprate=self.overlaprate),        # [B, 16, 102, 45]
            nn.MaxPool2d(2),          # [B, 16, 51, 22]
            GhostBlock_soft(64, 64,overlaprate=self.overlaprate),
            nn.MaxPool2d(2),          # [B, 32, 25, 11]
            GhostBlock_soft(64, 64,overlaprate=self.overlaprate),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.first_relu(self.first_bn(self.first_conv(x)))
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))

class GhostCNN_spec_hard(nn.Module):
    def __init__(self, num_classes=8,overlaprate = 0.5):
        super(GhostCNN_spec_hard, self).__init__()
        self.overlaprate = overlaprate
        self.first_conv = nn.Conv2d(1, 64, 3, 1)
        #self.first_bn = nn.BatchNorm2d(16)
        self.first_relu = nn.ReLU6(inplace=False)
        self.features = nn.Sequential(
            GhostBlock_hard(64, 64,overlaprate=self.overlaprate),        # [B, 16, 102, 45]
            nn.MaxPool2d(2),          # [B, 16, 51, 22]
            GhostBlock_hard(64, 64,overlaprate=self.overlaprate),
            nn.MaxPool2d(2),          # [B, 32, 25, 11]
            GhostBlock_hard(64, 64,overlaprate=self.overlaprate),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.first_relu(self.first_conv(x))
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


class MobileNetV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=6, stride=1):
        super(MobileNetV2Block, self).__init__()
        hidden = in_channels * expansion
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=False),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=False),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Identity() if in_channels == out_channels and stride == 1 else None

    def forward(self, x):
        out = self.block(x)
        if self.shortcut is not None:
            out += x
        return out


class MobileNetV2Block_soft(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=6, stride=1,overlaprate=0.5):
        super(MobileNetV2Block_soft, self).__init__()
        hidden = in_channels * expansion
        self.overlaprate=overlaprate
        #self.overlaprate = nn.Parameter(torch.tensor(-0.2))
        self.block = nn.Sequential(
            #nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            SegmentedSoftGroupConv(in_channels,hidden,1,1,num_groups=4,overlaprate_parameter=self.overlaprate),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=False),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=False),
            #nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            SegmentedSoftGroupConv(hidden,out_channels,1,1,num_groups=4,overlaprate_parameter=self.overlaprate),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Identity() if in_channels == out_channels and stride == 1 else None

    def forward(self, x):
        out = self.block(x)
        if self.shortcut is not None:
            out += x
        return out

class MobileNetV2Block_hard(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=6, stride=1,overlaprate = 0.5):
        super(MobileNetV2Block_hard, self).__init__()
        hidden = in_channels * expansion
        self.overlaprate = overlaprate
        self.block = nn.Sequential(
            #nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            OverlapConv2d(in_channels,hidden,1,1,num_groups=4,overlap_rate=self.overlaprate),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=False),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=False),
            OverlapConv2d(hidden,out_channels,1,1,num_groups=4,overlap_rate=self.overlaprate),
            #nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Identity() if in_channels == out_channels and stride == 1 else None

    def forward(self, x):
        out = self.block(x)
        if self.shortcut is not None:
            out += x
        return out

class MobileNetV2CNN_spec(nn.Module):
    def __init__(self, num_classes=8):
        super(MobileNetV2CNN_spec, self).__init__()
        self.first_conv = nn.Conv2d(1, 64, 3, 1)
        #self.first_bn = nn.BatchNorm2d(16)
        self.first_relu = nn.ReLU6(inplace=False)
        self.features = nn.Sequential(
            MobileNetV2Block(64, 64, expansion=1),
            nn.MaxPool2d(2),
            MobileNetV2Block(64, 64),
            nn.MaxPool2d(2),
            MobileNetV2Block(64, 64),
            # nn.MaxPool2d(2),
            # MobileNetV2Block(64, 64),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.first_relu(self.first_conv(x))
        #x = self.first_conv(x)
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))

class MobileNetV2CNN_spec_soft(nn.Module):
    def __init__(self, num_classes=8):
        super(MobileNetV2CNN_spec_soft, self).__init__()
        self.overlaprate = nn.Parameter(torch.tensor(-0.2))
        self.first_conv = nn.Conv2d(1,64,3,1)
        #self.first_bn = nn.BatchNorm2d(16)
        self.first_relu = nn.ReLU6(inplace=False)
        self.features = nn.Sequential(
            MobileNetV2Block_soft(64, 64, expansion=1,overlaprate=self.overlaprate),
            nn.MaxPool2d(2),
            MobileNetV2Block_soft(64, 64,overlaprate=self.overlaprate),
            nn.MaxPool2d(2),
            MobileNetV2Block_soft(64, 64,overlaprate=self.overlaprate),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.first_relu(self.first_conv(x))
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))

class MobileNetV2CNN_spec_hard(nn.Module):
    def __init__(self, num_classes=8,overlaprate=0.5):
        super(MobileNetV2CNN_spec_hard, self).__init__()
        self.overlaprate = overlaprate
        self.first_conv = nn.Conv2d(1, 64, 3, 1)
        #self.first_bn = nn.BatchNorm2d(16)
        self.first_relu = nn.ReLU6(inplace=False)
        self.features = nn.Sequential(
            MobileNetV2Block_hard(64, 64, expansion=1,overlaprate=self.overlaprate),
            nn.MaxPool2d(2),
            MobileNetV2Block_hard(64, 64,overlaprate=self.overlaprate),
            nn.MaxPool2d(2),
            MobileNetV2Block_hard(64, 64,overlaprate=self.overlaprate),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.first_relu(self.first_conv(x))
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))
class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1, expand3x3):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=False)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return self.relu(torch.cat([
            self.expand1x1(x),
            self.expand3x3(x)
        ], 1))

class Fire_soft(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1, expand3x3,overlaprate):
        super(Fire_soft, self).__init__()
        self.overlaprate = overlaprate
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        #self.squeeze = SegmentedSoftGroupConv(in_channels,squeeze_channels,1,1,num_groups=4,overlaprate_parameter=self.overlaprate)
        self.squeeze_activation = nn.ReLU(inplace=False)
        #self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1, kernel_size=1)
        self.expand1x1 = SegmentedSoftGroupConv(squeeze_channels,expand1x1,1,1,num_groups=4,overlaprate_parameter=self.overlaprate)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3, kernel_size=3, padding=1)
        #self.expand3x3 = SegmentedSoftGroupConv(squeeze_channels,expand3x3,3,1,num_groups=4,overlaprate_parameter=self.overlaprate)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return self.relu(torch.cat([
            self.expand1x1(x),
            self.expand3x3(x)
        ], 1))

class Fire_hard(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1, expand3x3,overlaprate):
        super(Fire_hard, self).__init__()
        self.overlaprate = overlaprate
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        #self.squeeze = OverlapConv2d(in_channels,squeeze_channels,1,1,num_groups=4,overlap_rate=self.overlaprate)
        self.squeeze_activation = nn.ReLU(inplace=False)
        self.expand1x1 = OverlapConv2d(squeeze_channels,expand1x1,1,1,num_groups=4,overlap_rate=self.overlaprate)
        #self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1, kernel_size=1)
        #self.expand1x1 = OverlapConv2d(squeeze_channels, expand3x3, 1, 1, num_groups=4, overlap_rate=self.overlaprate)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return self.relu(torch.cat([
            self.expand1x1(x),
            self.expand3x3(x)
        ], 1))

# class SqueezeCNN_spec(nn.Module):
#     def __init__(self, num_classes=8):
#         super(SqueezeCNN_spec, self).__init__()
#         self.first_conv = nn.Conv2d(1, 64, 3, 1)
#         #self.first_bn = nn.BatchNorm2d(16)
#         self.first_relu = nn.ReLU6(inplace=False)
#         self.features = nn.Sequential(
#             Fire(64, 16, 16, 16),       # 16 channels out
#             nn.MaxPool2d(2),
#             Fire(32, 16, 32, 32),   # 32 channels
#             nn.MaxPool2d(2),
#             Fire(64, 8, 32, 32),   # 32+16 = 48
#             #nn.Conv2d(128, 64, kernel_size=1),  # unify output
#             nn.AdaptiveAvgPool2d((1, 1))
#         )
#         self.classifier = nn.Linear(64, num_classes)
#
#     def forward(self, x):
#         x = self.first_relu(self.first_conv(x))
#         x = self.features(x)
#         return self.classifier(x.view(x.size(0), -1))
class SqueezeCNN_spec(nn.Module):
    def __init__(self, num_classes=8):
        super(SqueezeCNN_spec, self).__init__()
        self.first_conv = nn.Conv2d(1, 64, 3, 1)
        #self.first_bn = nn.BatchNorm2d(16)
        self.first_relu = nn.ReLU6(inplace=False)
        self.features = nn.Sequential(
            Fire(64, 32, 32, 32),       # 16 channels out
            nn.MaxPool2d(2),
            Fire(64, 32, 32, 32),   # 32 channels
            nn.MaxPool2d(2),
            Fire(64, 32, 32, 32),   # 32+16 = 48
            #nn.Conv2d(128, 64, kernel_size=1),  # unify output
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.first_relu(self.first_conv(x))
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))
class SqueezeCNN_spec_soft(nn.Module):
    def __init__(self, num_classes=8):
        super(SqueezeCNN_spec_soft, self).__init__()
        self.overlaprate = nn.Parameter(torch.tensor(-0.2))
        self.first_conv = nn.Conv2d(1, 16, 3, 1)
        self.first_bn = nn.BatchNorm2d(16)
        self.first_relu = nn.ReLU6(inplace=False)
        self.features = nn.Sequential(
            Fire_soft(16, 8, 16, 16,overlaprate=self.overlaprate),       # 16 channels out
            nn.MaxPool2d(2),
            Fire_soft(32, 8, 16, 16,overlaprate=self.overlaprate),   # 32 channels
            nn.MaxPool2d(2),
            Fire_soft(32, 8, 32, 32,overlaprate=self.overlaprate),   # 32+16 = 48
            #nn.Conv2d(48, 64, kernel_size=1),  # unify output
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.first_relu(self.first_bn(self.first_conv(x)))
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


class SqueezeCNN_spec_hard(nn.Module):
    def __init__(self, num_classes=8,overlaprate = 0.5):
        super(SqueezeCNN_spec_hard, self).__init__()
        self.overlaprate = overlaprate
        self.first_conv = nn.Conv2d(1, 64, 3, 1)
        #self.first_bn = nn.BatchNorm2d(16)
        self.first_relu = nn.ReLU6(inplace=False)
        self.features = nn.Sequential(
            Fire_hard(64, 32, 32, 32,overlaprate=self.overlaprate),       # 16 channels out
            nn.MaxPool2d(2),
            Fire_hard(64, 32, 32, 32,overlaprate=self.overlaprate),   # 32 channels
            nn.MaxPool2d(2),
            Fire_hard(64, 32, 32, 32,overlaprate=self.overlaprate),   # 32+16 = 48
            #nn.Conv2d(48, 64, kernel_size=1),  # unify output
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.first_relu(self.first_conv(x))
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))

# 简化的 ShuffleNetV2 block
class ShuffleNetV2Block(nn.Module):
    def __init__(self, channels):
        super(ShuffleNetV2Block, self).__init__()
        mid = channels // 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, 1, 1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat((x1, self.branch2(x2)), dim=1)
        return self.channel_shuffle(out)

    @staticmethod
    def channel_shuffle(x, groups=2):
        B, C, H, W = x.size()
        x = x.view(B, groups, C // groups, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(B, C, H, W)
class ShuffleNetV2DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShuffleNetV2DownBlock, self).__init__()
        mid = out_channels // 2
        assert out_channels % 2 == 0, "Output channels must be divisible by 2"

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, mid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, stride=2, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return ShuffleNetV2Block.channel_shuffle(out)



class ShuffleNetV2Block_soft(nn.Module):
    def __init__(self, channels,overlaprate):  # in_channels == out_channels
        super(ShuffleNetV2Block_soft, self).__init__()

        self.channels = channels
        mid = channels // 2
        self.overlaprate = overlaprate
        self.branch2 = nn.Sequential(
            #nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
            SegmentedSoftGroupConv(mid,mid,1,1,num_groups=4,overlaprate_parameter=self.overlaprate),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid, mid, 3, 1, 1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            #nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
            SegmentedSoftGroupConv(mid,mid,1,1,num_groups=4,overlaprate_parameter=self.overlaprate),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)  # split channels
        out = torch.cat((x1, self.branch2(x2)), dim=1)
        return self.channel_shuffle_soft(out)

    @staticmethod
    def channel_shuffle_soft(x, groups=2):
        N, C, H, W = x.size()
        x = x.view(N, groups, C // groups, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(N, C, H, W)


class ShuffleNetV2Block_hard(nn.Module):
    def __init__(self, channels,overlaprate):  # in_channels == out_channels
        super(ShuffleNetV2Block_hard, self).__init__()
        self.channels = channels
        mid = channels // 2
        self.overlaprate = overlaprate
        self.branch2 = nn.Sequential(
            #nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
            OverlapConv2d(mid,mid,1,1,num_groups=4,overlap_rate=self.overlaprate),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, 1, 1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            #nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
            OverlapConv2d(mid, mid, 1, 1, num_groups=4, overlap_rate=self.overlaprate),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)  # split channels
        out = torch.cat((x1, self.branch2(x2)), dim=1)
        return self.channel_shuffle_hard(out)

    @staticmethod
    def channel_shuffle_hard(x, groups=2):
        N, C, H, W = x.size()
        x = x.view(N, groups, C // groups, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(N, C, H, W)

class ShuffleNetV2DownBlock_soft(nn.Module):
    def __init__(self, in_channels, out_channels,overlaprate):
        super(ShuffleNetV2DownBlock_soft, self).__init__()
        mid = out_channels // 2
        assert out_channels % 2 == 0, "Output channels must be divisible by 2"
        self.overlaprate = overlaprate

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            #nn.Conv2d(in_channels, mid, 1, 1, 0, bias=False),
            SegmentedSoftGroupConv(in_channels,mid,1,1,num_groups=4,overlaprate_parameter=self.overlaprate),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            #nn.Conv2d(in_channels, mid, 1, 1, 0, bias=False),
            SegmentedSoftGroupConv(in_channels, mid, 1, 1, num_groups=4, overlaprate_parameter=self.overlaprate),

            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, stride=2, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            #nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
            SegmentedSoftGroupConv(mid, mid, 1, 1, num_groups=4, overlaprate_parameter=self.overlaprate),

            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return ShuffleNetV2Block_soft.channel_shuffle_soft(out)
class ShuffleNetV2DownBlock_hard(nn.Module):
    def __init__(self, in_channels, out_channels,overlaprate):
        super(ShuffleNetV2DownBlock_hard, self).__init__()
        mid = out_channels // 2
        assert out_channels % 2 == 0, "Output channels must be divisible by 2"
        self.overlaprate = overlaprate

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            #nn.Conv2d(in_channels, mid, 1, 1, 0, bias=False),
            OverlapConv2d(in_channels,mid,1,1,num_groups=4,overlap_rate=self.overlaprate),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            #nn.Conv2d(in_channels, mid, 1, 1, 0, bias=False),
            OverlapConv2d(in_channels, mid, 1, 1, num_groups=4, overlap_rate=self.overlaprate),

            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, stride=2, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            #nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
            OverlapConv2d(mid, mid, 1, 1, num_groups=4, overlap_rate=self.overlaprate),

            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return ShuffleNetV2Block_hard.channel_shuffle_hard(out)

class ShuffleCNN_spec(nn.Module):
    def __init__(self, num_classes=8):
        super(ShuffleCNN_spec, self).__init__()
        self.first_conv = nn.Conv2d(1, 64, 3, 1)
        #self.first_bn = nn.BatchNorm2d(16)
        self.first_relu = nn.ReLU6(inplace=False)
        self.features = nn.Sequential(
            ShuffleNetV2Block(64, 64),    # 1 -> 16
            nn.MaxPool2d(2),
            ShuffleNetV2Block(64, 64),
            nn.MaxPool2d(2),
            ShuffleNetV2Block(64, 64),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64 + 64, num_classes)  # concat channel count

    def forward(self, x):
        x = self.first_relu(self.first_conv(x))
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))
class ShuffleNetV2CNN_spec(nn.Module):
    def __init__(self, num_classes=8):
        super(ShuffleNetV2CNN_spec, self).__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.features = nn.Sequential(
            ShuffleNetV2Block(64),    # 下采样 + 扩通道
            nn.MaxPool2d(2),
            ShuffleNetV2Block(64),            # 保持通道
            nn.MaxPool2d(2),
            ShuffleNetV2Block(64),    # 下采样 + 扩通道
            #ShuffleNetV2Block(64),            # 保持通道
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))

class ShuffleNetV2CNN_spec_soft(nn.Module):
    def __init__(self, num_classes=8):
        super(ShuffleNetV2CNN_spec_soft, self).__init__()
        self.overlaprate = nn.Parameter(torch.tensor(-0.2))
        # self.first_conv = nn.Conv2d(1, 16, 3, 1)
        # self.first_bn = nn.BatchNorm2d(16)
        # self.first_relu = nn.ReLU6(inplace=False)
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.features = nn.Sequential(
            ShuffleNetV2DownBlock_soft(16, 32,overlaprate=self.overlaprate),    # 下采样 + 扩通道
            ShuffleNetV2Block_soft(32,overlaprate=self.overlaprate),            # 保持通道
            ShuffleNetV2DownBlock_soft(32, 64,overlaprate=self.overlaprate),    # 下采样 + 扩通道
            #ShuffleNetV2Block(64),            # 保持通道
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))
class ShuffleNetV2CNN_spec_hard(nn.Module):
    def __init__(self, num_classes=8,overlaprate = 0.5):
        super(ShuffleNetV2CNN_spec_hard, self).__init__()
        self.overlaprate = overlaprate
        # self.first_conv = nn.Conv2d(1, 16, 3, 1)
        # self.first_bn = nn.BatchNorm2d(16)
        # self.first_relu = nn.ReLU6(inplace=False)
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.features = nn.Sequential(
            ShuffleNetV2Block_hard(64,overlaprate=self.overlaprate),     # 下采样 + 扩通道
            nn.MaxPool2d(2),
            ShuffleNetV2Block_hard(64,overlaprate=self.overlaprate),            # 保持通道
            nn.MaxPool2d(2),
            ShuffleNetV2Block_hard(64,overlaprate=self.overlaprate),     # 下采样 + 扩通道
            #ShuffleNetV2Block(64),            # 保持通道
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, scale=3):
        super(Bottle2neck, self).__init__()
        self.groups = planes // scale
        self.len = planes // scale
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)

        self.conv2 = nn.Conv2d(planes // scale, planes // scale, kernel_size=3, groups=planes // scale, padding=1)
        self.conv3 = nn.Conv2d(planes // scale, planes // scale, kernel_size=1, stride=1)
        self.scale = scale
        self.shortcut = None
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride),
            )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        # out =shuffle_chnls(out, self.groups)
        spx = torch.split(out, self.len, 1)  # 从通道的维度切割,待分输入，需要切分的大小，切分维度
        side = self.conv2(spx[1])
        side = self.relu(side)
        side = self.conv3(side)
        z = torch.cat((spx[0], side), 1)
        for i in range(2, self.scale):
            sp = side + spx[i]
            y = self.conv2(sp)
            y = self.relu(y)
            y = self.conv3(y)
            side = y
            z = torch.cat((z, y), 1)
        out = z
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = out + residual
        out = self.relu(out)
        return out

class Bottle2neck_soft(nn.Module):

    def __init__(self, inplanes, planes, overlaprate, stride=1, scale=3 ):
        super(Bottle2neck_soft, self).__init__()
        self.groups = planes // scale
        self.len = planes // scale
        self.overlap = overlaprate
        # self.overlaprate = nn.Parameter(torch.tensor(-0.2))
        #self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        if inplanes>18:
            self.conv1 = SegmentedSoftGroupConv(inplanes, planes, 1, stride, num_groups=3,
                                                 overlaprate_parameter=self.overlap)
        else:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.conv2 = nn.Conv2d(planes // scale, planes // scale, kernel_size=3, groups=planes // scale, padding=1)
        # if planes // scale<3:
        #     self.conv3 = nn.Conv2d(planes//scale, planes//scale, kernel_size=1, stride=1)
        # else:
        #     self.conv3 = SegmentedSoftGroupConv(planes // scale, planes // scale, 1, 1, num_groups=3,
        #                                         overlaprate_parameter=self.overlap)
        self.conv3 = nn.Conv2d(planes // scale, planes // scale, kernel_size=1, stride=1)
        #self.conv3 = SegmentedSoftGroupConv(planes // scale, planes // scale, 1, 1, num_groups=3,
        #                                   overlaprate_parameter=self.overlap)
        self.scale = scale
        self.shortcut = None
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride),
            )
        # if stride != 1 or inplanes != planes:
        #     self.shortcut = nn.Sequential(
        #         SegmentedSoftGroupConv(inplanes, planes, 1, stride, num_groups=3, overlaprate_parameter=self.overlap)
        #     )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        # out =shuffle_chnls(out, self.groups)
        spx = torch.split(out, self.len, 1)  # 从通道的维度切割,待分输入，需要切分的大小，切分维度
        side = self.conv2(spx[1])
        side = self.relu(side)
        side = self.conv3(side)
        z = torch.cat((spx[0], side), 1)
        for i in range(2, self.scale):
            sp = side + spx[i]
            y = self.conv2(sp)
            y = self.relu(y)
            y = self.conv3(y)
            side = y
            z = torch.cat((z, y), 1)
        out = z
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = out + residual
        out = self.relu(out)
        return out


class Bottle2neck_hard(nn.Module):

    def __init__(self, inplanes, planes, overlaprate, stride=1, scale=3):
        super(Bottle2neck_hard, self).__init__()
        self.groups = planes // scale
        self.len = planes // scale
        self.overlaprate = overlaprate
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.conv1 = OverlapConv2d(inplanes, planes, 1, stride, num_groups=4, overlap_rate=self.overlaprate)

        self.conv2 = nn.Conv2d(planes // scale, planes // scale, kernel_size=3, groups=planes // scale, padding=1)
        # self.conv3 = nn.Conv2d(planes // scale, planes // scale, kernel_size=1, stride=1)
        self.conv3 = OverlapConv2d(planes // scale, planes // scale, 1, 1, num_groups=4, overlap_rate=self.overlaprate)
        self.scale = scale
        self.shortcut = None
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride),
            )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        # out =shuffle_chnls(out, self.groups)
        spx = torch.split(out, self.len, 1)  # 从通道的维度切割,待分输入，需要切分的大小，切分维度
        side = self.conv2(spx[1])
        side = self.relu(side)
        side = self.conv3(side)
        z = torch.cat((spx[0], side), 1)
        for i in range(2, self.scale):
            sp = side + spx[i]
            y = self.conv2(sp)
            y = self.relu(y)
            y = self.conv3(y)
            side = y
            z = torch.cat((z, y), 1)
        out = z
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = out + residual
        out = self.relu(out)
        return out


class LMSC_spec_hard(nn.Module):
    def __init__(self, num_classes=8,overlaprate = 0.5):
        super(LMSC_spec_hard, self).__init__()
        self.overlaprate = overlaprate
        # self.first_conv = nn.Conv2d(1, 16, 3, 1)
        # self.first_bn = nn.BatchNorm2d(16)
        # self.first_relu = nn.ReLU6(inplace=False)
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, 96, 3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.features = nn.Sequential(
            Bottle2neck_hard(96,96,overlaprate=self.overlaprate),     # 下采样 + 扩通道
            nn.MaxPool2d(2),
            Bottle2neck_hard(96,96,overlaprate=self.overlaprate),           # 保持通道
            nn.MaxPool2d(2),
            Bottle2neck_hard(96,96,overlaprate=self.overlaprate),     # 下采样 + 扩通道
            #ShuffleNetV2Block(64),            # 保持通道
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))

class LMSC_spec(nn.Module):
    def __init__(self, num_classes=8):
        super(LMSC_spec, self).__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, 96, 3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.features = nn.Sequential(
            Bottle2neck(96,96),    # 下采样 + 扩通道
            nn.MaxPool2d(2),
            Bottle2neck(96,96),            # 保持通道
            nn.MaxPool2d(2),
            Bottle2neck(96,96),   # 下采样 + 扩通道
            #ShuffleNetV2Block(64),            # 保持通道
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))