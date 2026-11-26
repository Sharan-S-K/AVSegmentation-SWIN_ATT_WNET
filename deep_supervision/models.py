import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNetDeepSup(nn.Module):


    def __init__(self, in_channels=3, out_channels=2, filters=[32, 64, 128, 256, 512], dropout=0.0):
        super(AttentionUNetDeepSup, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


        self.Conv1 = ConvBlock(in_channels, filters[0], dropout)
        self.Conv2 = ConvBlock(filters[0], filters[1], dropout)
        self.Conv3 = ConvBlock(filters[1], filters[2], dropout)
        self.Conv4 = ConvBlock(filters[2], filters[3], dropout)
        self.Conv5 = ConvBlock(filters[3], filters[4], dropout)


        self.Up5 = UpConv(filters[4], filters[3])
        self.Att5 = AttentionBlock(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = ConvBlock(filters[4], filters[3], dropout)

        self.Up4 = UpConv(filters[3], filters[2])
        self.Att4 = AttentionBlock(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = ConvBlock(filters[3], filters[2], dropout)

        self.Up3 = UpConv(filters[2], filters[1])
        self.Att3 = AttentionBlock(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = ConvBlock(filters[2], filters[1], dropout)

        self.Up2 = UpConv(filters[1], filters[0])
        self.Att2 = AttentionBlock(F_g=filters[0], F_l=filters[0], F_int=filters[0]//2)
        self.Up_conv2 = ConvBlock(filters[1], filters[0], dropout)


        self.Conv_1x1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)


        self.ds_conv4 = nn.Conv2d(filters[3], out_channels, kernel_size=1)
        self.ds_conv3 = nn.Conv2d(filters[2], out_channels, kernel_size=1)
        self.ds_conv2 = nn.Conv2d(filters[1], out_channels, kernel_size=1)

    def forward(self, x, return_deep_sup=True):
        input_size = x.shape[2:]


        e1 = self.Conv1(x)
        e2 = self.Conv2(self.Maxpool(e1))
        e3 = self.Conv3(self.Maxpool(e2))
        e4 = self.Conv4(self.Maxpool(e3))
        e5 = self.Conv5(self.Maxpool(e4))


        d5 = self.Up5(e5)
        e4_att = self.Att5(g=d5, x=e4)
        d5 = torch.cat((e4_att, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        e3_att = self.Att4(g=d4, x=e3)
        d4 = torch.cat((e3_att, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        e2_att = self.Att3(g=d3, x=e2)
        d3 = torch.cat((e2_att, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        e1_att = self.Att2(g=d2, x=e1)
        d2 = torch.cat((e1_att, d2), dim=1)
        d2 = self.Up_conv2(d2)


        out = self.Conv_1x1(d2)

        if return_deep_sup:

            ds4 = F.interpolate(self.ds_conv4(d5), size=input_size, mode='bilinear', align_corners=True)
            ds3 = F.interpolate(self.ds_conv3(d4), size=input_size, mode='bilinear', align_corners=True)
            ds2 = F.interpolate(self.ds_conv2(d3), size=input_size, mode='bilinear', align_corners=True)
            return out, [ds4, ds3, ds2]

        return out


class AttentionWNetDeepSup(nn.Module):


    def __init__(self, in_channels=3, out_channels1=2, out_channels2=3, filters=[32, 64, 128, 256, 512], dropout=0.0):
        super(AttentionWNetDeepSup, self).__init__()
        self.unet1 = AttentionUNetDeepSup(in_channels=in_channels, out_channels=out_channels1, filters=filters, dropout=dropout)
        self.unet2 = AttentionUNetDeepSup(in_channels=in_channels + out_channels1, out_channels=out_channels2, filters=filters, dropout=dropout)

    def forward(self, x, return_deep_sup=True):
        if x.ndim != 4 or x.shape[-2] % 16 or x.shape[-1] % 16 or min(x.shape[-2:]) < 16:
            raise ValueError("Expected BCHW input with height and width divisible by 16")
        if return_deep_sup:
            out1_logits, ds1_list = self.unet1(x, return_deep_sup=True)
        else:
            out1_logits = self.unet1(x, return_deep_sup=False)
            ds1_list = None

        out1_probs = torch.softmax(out1_logits, dim=1)

        x2 = torch.cat([x, out1_probs], dim=1)

        if return_deep_sup:
            out2_logits, ds2_list = self.unet2(x2, return_deep_sup=True)
        else:
            out2_logits = self.unet2(x2, return_deep_sup=False)
            ds2_list = None

        out2_probs = torch.softmax(out2_logits, dim=1)

        if return_deep_sup:

            ds1_probs = [torch.softmax(ds, dim=1) for ds in ds1_list]
            ds2_probs = [torch.softmax(ds, dim=1) for ds in ds2_list]
            return out1_probs, out2_probs, ds1_probs, ds2_probs

        return out1_probs, out2_probs


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, filters=[32, 64, 128, 256, 512], dropout=0.0):
        super(AttentionUNet, self).__init__()
        self.model = AttentionUNetDeepSup(in_channels, out_channels, filters, dropout)

    def forward(self, x):
        return self.model(x, return_deep_sup=False)


class AttentionWNet(nn.Module):
    def __init__(self, in_channels=3, out_channels1=2, out_channels2=3, filters=[32, 64, 128, 256, 512], dropout=0.0):
        super(AttentionWNet, self).__init__()
        self.model = AttentionWNetDeepSup(in_channels, out_channels1, out_channels2, filters, dropout)

    def forward(self, x):
        return self.model(x, return_deep_sup=False)
