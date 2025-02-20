import torch
import torch.nn as nn
from SimSiam import Distance
from torchvision import transforms
from PIL import Image
from Image_enhancement import figure_rand

hid = 32
growthRate = 16
class PTrans(nn.Module):
    def __init__(self, in_ch):
        super(PTrans, self).__init__()
        self.q_conv = nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1)
        self.k_conv = nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1)
        self.softmax = nn.Softmax(-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        D, C, H, W = x.size()
        q = self.q_conv(x).view(D, -1, W * H).permute(0, 2, 1)
        k = self.k_conv(x).view(D, -1, W * H)
        v = x.view(D, -1, H*W)
        attention = torch.bmm(q, k)
        attention = self.softmax(attention)
        out = torch.bmm(v, attention.permute(0, 2, 1)).view(D, C, H, W)
        out = self.gamma*out + x
        return out, attention


class CTrans(nn.Module):
    def __init__(self):
        super(CTrans, self).__init__()
        # self.softmax = softmax1(axis=-1)
        self.softmax = nn.Softmax(-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        D, C, H, W = x.size()
        q = x.view(D, C, -1)
        k = x.view(D, C, -1).permute(0, 2, 1)
        v = x.view(D, C, H * W)
        energy = torch.bmm(q, k)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        out = torch.bmm(attention.permute(0, 2, 1), v).view(D, C, H, W)
        out = self.gamma * out + x
        return out, attention

class ConvLayer(nn.Module):
    def __init__(self, in_feature, out_feature, k, pad):
        super(ConvLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_feature)
        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=k, stride=1, padding=pad)
        self.GELU = nn.GELU()

    def forward(self, x):
        out = self.bn(x)
        out = self.GELU(out)
        out = self.conv(out)

        return out

class DenseBlock(nn.Module):
    """
        深度卷积模块
        nChannels = 输入通道数
        growthRate = ConvLayer模块输出通道数
        nDenseBlocks ：深度卷积层数
    """
    def __init__(self, nChannels, growthRate, nDenseBlocks, k, pad):
        super().__init__()
        self.layers = self.make_dense(nChannels, growthRate, nDenseBlocks, k, pad)

    def make_dense(self, nChannels, growthRate, nDenseBlocks, k, pad):
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(ConvLayer(nChannels, growthRate, k, pad))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        outtemp = x
        for layer in self.layers[:]:
            out = layer(outtemp)
            outtemp = torch.cat([outtemp, out], dim=1)
        out = outtemp
        return out

class base_module(nn.Module):
    def __init__(self, hsi_ch, nDenseBlocks, growthRate, hidden_ch, out_ch, num_layers=1, p=0.1):
        super().__init__()
        self.conv = nn.Conv2d(hsi_ch, hidden_ch, kernel_size=1, stride=1, padding=0)
        self.SpectralBlock_1 = DenseBlock(hidden_ch, growthRate, nDenseBlocks, (7, 7), (3, 3))
        self.SpectralBlock_2 = DenseBlock(hidden_ch, growthRate, nDenseBlocks, (5, 5), (2, 2))
        self.SpectralBlock_3 = DenseBlock(hidden_ch, growthRate, nDenseBlocks, (3, 3), (1, 1))
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d((hidden_ch + nDenseBlocks*growthRate) * 3),
            nn.GELU(),
            nn.Conv2d((hidden_ch + nDenseBlocks * growthRate) * 3, out_ch, kernel_size=(1, 1),stride=1, padding=0)
        )

        self.ctran = nn.Sequential(
            CTrans(),
        )
        self.ptran = nn.Sequential(
            PTrans(out_ch*2),
        )
        self.output_dim = out_ch*2
        self.dropout = nn.Dropout(p)
        self.num_layers = num_layers

    def forward(self, x):
        x = self.conv(x)
        x1 = self.SpectralBlock_1(x)
        x2 = self.SpectralBlock_2(x)
        x3 = self.SpectralBlock_3(x)
        out = torch.cat([x1, x2], dim=1)
        out = torch.cat([out, x3], dim=1)
        out = self.conv1(out)

        return out


class prediction_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):  # bottleneck structure
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SCE_fusion(nn.Module):
    def __init__(self, hsi_ch, msi_ch, insize, hidden_ch, out_ch):
        super().__init__()

        self.figure_rand = figure_rand()
        self.backbone2 = base_module(msi_ch, 3, growthRate=growthRate, hidden_ch=hidden_ch//2, out_ch=out_ch, num_layers=2)
        self.backbone3 = base_module(hsi_ch, 3, growthRate=growthRate, hidden_ch=hidden_ch//2, out_ch=out_ch, num_layers=3)
        self.projector2 = nn.Linear(insize * insize, 1)
        self.projector3 = nn.Linear(insize * insize//16, 1)
        self.mask = nn.Dropout2d(0.5)
        self.ctran2 = nn.Sequential(
            CTrans(),
        )
        self.ptran = nn.Sequential(
            PTrans(out_ch),
        )
        self.predictor = prediction_MLP(out_ch*4, 256, out_ch*4)
        self.predictor2 = prediction_MLP(out_ch, 128, out_ch)
        self.hsi_ch = hsi_ch
        self.insize = insize

        self.criteon = nn.L1Loss()

    def forward(self, hsi, msi):
        hsi1 = hsi
        hsi2 = self.figure_rand(hsi)
        msi1 = msi
        msi2 = self.figure_rand(msi)

        f2, f3 = self.backbone2, self.backbone3

        fx4 = f2(msi1)
        fx5 = f2(msi2)

        fx7 = f3(hsi1)
        fx8 = f3(hsi2)

        ax4, atten4 = self.ptran(fx4)
        ax5, atten5 = self.ptran(fx5)
        D, C, H, W = ax4.size()
        x4 = ax4.view(D, C, H * W)
        x4 = self.projector2(x4).squeeze(2)
        x5 = ax5.view(D, C, H * W)
        x5 = self.projector2(x5).squeeze(2)

        ax7, atten7 = self.ctran2(fx7)
        ax8, atten8 = self.ctran2(fx8)
        D, C, H, W = ax7.size()
        x7 = ax7.view(D, C, H * W)
        x7 = self.projector3(x7).squeeze(2)
        x8 = ax8.view(D, C, H * W)
        x8 = self.projector3(x8).squeeze(2)

        pred4 = self.predictor2(x4)
        pred5 = self.predictor2(x5)

        pred7 = self.predictor2(x7)
        pred8 = self.predictor2(x8)

        L2 = (Distance(pred4, x5) + Distance(pred5, x4)) / 2
        L3 = (Distance(pred7, x8) + Distance(pred8, x7)) / 2

        cl_loss = (L2 * 0.5 + L3 * 0.5)

        return cl_loss, fx7, atten7, fx4, atten4



class PTrans2(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.q_conv = nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1)
        self.k_conv = nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1)
        self.softmax = nn.Softmax(-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, atten):
        D, C, H, W = x.size()
        q = self.q_conv(x).view(D, -1, W * H).permute(0, 2, 1)
        k = self.k_conv(x).view(D, -1, W * H)
        v = x.view(D, -1, H*W)
        attention = torch.bmm(q, k)
        attention = self.softmax(attention)
        attention = self.softmax(attention + atten)
        out = torch.bmm(v, attention.permute(0, 2, 1)).view(D, C, H, W)
        out = self.gamma*out + x
        return out

class CTrans2(nn.Module):
    def __init__(self):
        super().__init__()
        # self.softmax = softmax1(axis=-1)
        self.softmax = nn.Softmax(-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, atten):
        D, C, H, W = x.size()
        q = x.view(D, C, -1)
        k = x.view(D, C, -1).permute(0, 2, 1)
        v = x.view(D, C, H * W)
        energy = torch.bmm(q, k)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        attention = self.softmax(attention + atten)
        out = torch.bmm(attention.permute(0, 2, 1), v).view(D, C, H, W)
        out = self.gamma * out + x
        return out

class AGCF(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.ptran = PTrans2(in_ch)
        self.ctran = CTrans2()

    def forward(self, x, atten1, atten2):
        x = self.ptran(x, atten1)
        x = self.ctran(x, atten2)

        return x


class DownBlock(nn.Module):
    """
        in_ch:输入维度
        x：输入batch
    """
    def __init__(self, in_ch, hid = hid):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, in_ch + hid, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(in_ch + hid)
        self.GELU = nn.GELU()
        self.conv2 = nn.Conv2d(in_ch + hid, in_ch + hid, kernel_size=3, padding=1, stride=1)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.GELU(self.bn(x))
        x = self.conv2(x)
        out = self.maxpool(x)
        return x, out

class UpBlock(nn.Module):
    """
        in_ch:输入维度
        x：本层输出
        in_img：上层输入
    """
    def __init__(self, in_ch):
        super(UpBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2, padding=0)
        self.bn = nn.BatchNorm2d(in_ch//2)
        self.GELU = nn.GELU()
        self.conv1 = nn.Conv2d(in_ch, in_ch//2, kernel_size=3, stride=1, padding=1)

    def forward(self, x, in_img):
        x = self.deconv(x)
        x = self.GELU(self.bn(x))
        x = torch.cat([x, in_img], dim=1)
        x = self.conv1(x)
        return x

class MSSF(nn.Module):
    def __init__(self, in_ch, hsi_ch, M_hid=32, hid=hid, U_hid=64):
        super().__init__()


        self.down1 = DownBlock(in_ch)
        self.down2 = DownBlock(in_ch + hid)
        self.down3 = DownBlock(in_ch + hid * 2)
        self.Mhid = M_hid
        #第3层
        self.c4 = nn.Conv2d(in_ch + hid * 3 ,in_ch + hid * 4, kernel_size=3, stride=1, padding=1)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(in_ch + hid * 4, in_ch + hid * 4, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(in_ch + hid * 4),
            nn.GELU(),
            nn.Conv2d(in_ch + hid * 4, M_hid, kernel_size=3, stride=1, padding=1)
        )
        self.m3_3 = nn.Conv2d(in_ch + hid * 3, M_hid, kernel_size=3, stride=1, padding=1)
        self.m3_2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_ch + hid * 2, M_hid, kernel_size=3, stride=1, padding=1)
        )
        self.m3_1 = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(in_ch + hid, M_hid, kernel_size=3, stride=1, padding=1)
        )
        self.out3 = nn.Conv2d(M_hid*4, U_hid*3, kernel_size=3, stride=1, padding=1)
        #第2层
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(U_hid*3, U_hid*3, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(U_hid*3),
            nn.GELU(),
            nn.Conv2d(U_hid*3, M_hid, kernel_size=3, stride=1, padding=1)
        )
        self.m2_2 = nn.Conv2d(in_ch + hid * 2, M_hid, kernel_size=3, stride=1, padding=1)
        self.m2_1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_ch + hid * 1, M_hid, kernel_size=3, stride=1, padding=1)
        )
        self.out2 = nn.Conv2d(M_hid * 3, U_hid * 2, kernel_size=3, stride=1, padding=1)
        #第一层
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(U_hid*2, U_hid*2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(U_hid*2),
            nn.GELU(),
            nn.Conv2d(U_hid*2, M_hid, kernel_size=3, stride=1, padding=1)
        )
        self.m1_1 = nn.Conv2d(in_ch + hid, M_hid, kernel_size=3, stride=1, padding=1)
        self.out1 = nn.Conv2d(M_hid * 2, hsi_ch, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x1, d2 = self.down1(x)
        x2, d3 = self.down2(d2)
        x3, d4 = self.down3(d3)
        x4 = self.c4(d4)
        x3_4 = self.up4(x4)
        x3_3 = self.m3_3(x3)
        x3_2 = self.m3_2(x2)
        x3_1 = self.m3_1(x1)
        u3 = self.out3(torch.cat([x3_4, x3_3, x3_2, x3_1], dim=1))
        x2_3 = self.up3(u3)
        x2_2 = self.m2_2(x2)
        x2_1 = self.m2_1(x1)
        u2 = self.out2(torch.cat([x2_3, x2_2, x2_1], dim=1))
        x1_2 = self.up2(u2)
        x1_1 = self.m1_1(x1)
        u1 = self.out1(torch.cat([x1_2, x1_1], dim=1))

        return u1


class AMFS(nn.Module):
    def __init__(self, hsi_ch, insize, hid_ch):
        super(AMFS, self).__init__()
        self.bic4 = transforms.Resize(size=(insize, insize), interpolation=Image.BICUBIC)
        self.conv = nn.Sequential(
            nn.Conv2d(hid_ch*2, hid_ch, kernel_size=3, padding=1, stride=1),
            nn.GELU()
        )
        self.clagmodel1 = AGCF(hid_ch)
        self.mulu = MSSF(hid_ch, hsi_ch)
        self.criteon = nn.L1Loss()

    def forward(self, fx7, atten7, fx4, atten4, gt):
        hsi = self.bic4(fx7)
        msi = fx4
        hr = self.conv(torch.cat([hsi,msi], dim=1))
        hr = self.clagmodel1(hr, atten4, atten7)
        hr = self.mulu(hr)
        L = self.criteon(hr, gt)

        return hr, L



class avg_center(nn.Module):
    def __init__(self, hsi_ch, hidden_ch1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(hsi_ch*2, hsi_ch, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Sequential(
            nn.Conv2d(hsi_ch, hidden_ch1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_ch1),
            nn.GELU()
        )

    def forward(self, x):
        D, C, W, H = x.size()
        x1 = x[:, :, W//2, H//2].view(D, C, 1, 1)
        x2 = self.avg(x)
        x_ = torch.cat([x1, x2], dim=1)
        x_ = self.conv(x_)
        out = x_ * x
        out = out + x
        out = self.conv1(out)
        return out


class SCE_classification(nn.Module):
    def __init__(self, hsi_ch, msi_ch, insize, hidden_ch, out_ch):
        super().__init__()
        self.figure_rand = figure_rand()
        self.backbone2 = base_module(msi_ch, 3, growthRate=growthRate, hidden_ch=hidden_ch//2, out_ch=out_ch, num_layers=2)
        self.backbone3 = base_module(hsi_ch, 3, growthRate=growthRate, hidden_ch=hidden_ch//2, out_ch=out_ch, num_layers=3)
        self.projector2 = nn.Linear(insize * insize, 1)
        self.projector3 = nn.Linear(insize ** 2//16, 1)
        self.mask = nn.Dropout2d(0.5)
        self.ctran2 = nn.Sequential(
            CTrans(),
        )
        self.ptran = nn.Sequential(
            PTrans(out_ch),
        )
        self.predictor = prediction_MLP(out_ch*4, 256, out_ch*4)
        self.predictor2 = prediction_MLP(out_ch, 128, out_ch)
        self.hsi_ch = hsi_ch
        self.insize = insize
        self.criteon = nn.L1Loss()

    def forward(self, hsi, msi):
        hsi1 = hsi
        hsi2 = self.figure_rand(hsi)
        msi1 = msi
        msi2 = self.figure_rand(msi)

        f2, f3 = self.backbone2, self.backbone3

        fx4 = f2(msi1)
        fx5 = f2(msi2)
        fx6 = f2(msi1)

        fx7 = f3(hsi1)
        fx8 = f3(hsi2)
        fx9 = f3(hsi1)

        g2, g3 = self.projector2, self.projector3
        ax4, atten4 = self.ptran(fx4)
        ax5, atten5 = self.ptran(fx5)
        ax6, atten6 = self.ptran(fx6)
        D, C, H, W = ax4.size()
        x4 = ax4.view(D, C, H * W)
        x4 = g2(x4).squeeze(2)
        x5 = ax5.view(D, C, H * W)
        x5 = g2(x5).squeeze(2)
        x6 = ax6.view(D, C, H * W)
        x6 = g2(x6).squeeze(2)

        ax7, atten7 = self.ctran2(fx7)
        ax8, atten8 = self.ctran2(fx8)
        ax9, atten9 = self.ctran2(fx9)
        D, C, H, W = ax7.size()
        x7 = ax7.view(D, C, H * W)
        x7 = g3(x7).squeeze(2)
        x8 = ax8.view(D, C, H * W)
        x8 = g3(x8).squeeze(2)
        x9 = ax9.view(D, C, H * W)
        x9 = g3(x9).squeeze(2)

        pred4 = self.predictor2(x4)
        pred5 = self.predictor2(x5)

        pred7 = self.predictor2(x7)
        pred8 = self.predictor2(x8)

        L2 = (Distance(pred4, x5) + Distance(pred5, x4)) / 2
        L3 = (Distance(pred7, x8) + Distance(pred8, x7)) / 2

        cl_loss = (L2 * 0.5 + L3 * 0.5)

        return cl_loss, fx7, atten7, fx4, atten4, x6, x9


class TI_CCS(nn.Module):
    def __init__(self, hsi_ch, insize, hidden_ch, in_ch, out_ch, p=0.1):
        super().__init__()
        self.figure_rand = figure_rand()
        self.backbone1 = base_module(hsi_ch, 3, growthRate=growthRate, hidden_ch=hidden_ch, out_ch=in_ch, num_layers=1)
        self.projector1 = nn.Linear(insize * insize, 1)
        self.projector2 = nn.Linear(in_ch * 2, in_ch)
        self.predictor1 = prediction_MLP(in_ch, 128, in_ch)
        self.linear = nn.Sequential(
                    nn.Linear(in_ch, 128),
                    nn.Dropout(p),
                    nn.GELU(),
                    nn.Linear(128, out_ch)
                    # nn.Softmax()
                )
        self.ctran = nn.Sequential(
            CTrans(),
        )
        self.ptran = nn.Sequential(
            PTrans(in_ch),
        )

    def forward(self, hr, x6, x9):
        hr1 = hr
        hr2 = self.figure_rand(hr)

        f1 = self.backbone1
        fx1 = f1(hr1)
        fx2 = f1(hr2)

        ax1, atten1 = self.ctran(fx1)
        ax2, atten2 = self.ptran(fx1)
        ax3, atten3 = self.ctran(fx2)
        ax4, atten4 = self.ptran(fx2)

        x1 = torch.cat([ax1, ax2], dim=1)
        x2 = torch.cat([ax3, ax4], dim=1)

        D, C, H, W = x1.size()
        x1 = x1.view(D, C, H * W)
        x1 = self.projector1(x1).squeeze(2)
        x1 = self.projector2(x1)
        x2 = x2.view(D, C, H * W)
        x2 = self.projector1(x2).squeeze(2)
        x2 = self.projector2(x2)

        pred1 = self.predictor1(x1)
        pred2 = self.predictor1(x2)

        L1 = (Distance(pred1, x2) + Distance(pred1, x6) + Distance(pred1, x9)) / 3
        L2 = (Distance(pred2, x1) + Distance(pred2, x6) + Distance(pred2, x9)) / 3

        L = (L1 + L2)/2

        out = self.linear(x1)
        return out, L















