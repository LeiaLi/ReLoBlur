import torch.nn.functional as F
from hparams import hparams
from .layers import *


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8, norm=False):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel, norm=norm) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8, norm=False):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel, norm=norm) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane - 3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class LBAG(nn.Module):
    def __init__(self, num_res=20, norm=False):
        super(LBAG, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res, norm=norm),
            EBlock(base_channel * 2, num_res, norm=norm),
            EBlock(base_channel * 4, num_res, norm=norm),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2, norm=norm),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2, norm=norm),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True,
                      norm=norm),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True, norm=norm),
            BasicConv(base_channel, 4, kernel_size=3, relu=False, stride=1, norm=norm)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res, norm=norm),
            DBlock(base_channel * 2, num_res, norm=norm),
            DBlock(base_channel, num_res, norm=norm)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1, norm=norm),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1, norm=norm),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 4, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 4, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel * 1),
            AFF(base_channel * 7, base_channel * 2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        gate_xs = []

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        res2 = self.drop2(res2)
        res1 = self.drop1(res1)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        if hparams.get('multiscale_gate', True):
            gate_x = F.sigmoid(z_[:, :1])
            if hparams['clamp_gate'] > 0:
                gate_x = gate_x.clamp_min(hparams['clamp_gate'])
            gate_xs.append(gate_x)
            outputs.append(z_[:, 1:] * gate_x + x_4)
        else:
            outputs.append(z_[:, 1:] + x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        if hparams.get('multiscale_gate', True):
            gate_x = F.sigmoid(z_[:, :1])
            if hparams['clamp_gate'] > 0:
                gate_x = gate_x.clamp_min(hparams['clamp_gate'])
            gate_xs.append(gate_x)
            outputs.append(z_[:, 1:] * gate_x + x_2)
        else:
            outputs.append(z_[:, 1:] + x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)

        gate_x = F.sigmoid(z[:, :1])
        if hparams['clamp_gate'] > 0:
            gate_x = gate_x.clamp_min(hparams['clamp_gate'])
        outputs.append(z[:, 1:] * gate_x + x)
        gate_xs.append(gate_x)

        return outputs, gate_xs
