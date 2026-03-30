import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.skip = nn.Identity()
        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class StrictBranch(nn.Module):
    def __init__(self, in_channels=1, base_ch=64):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.enc0 = ResidualBlock(in_channels, base_ch)
        self.enc1 = ResidualBlock(base_ch, base_ch*2)
        self.enc2 = ResidualBlock(base_ch*2, base_ch*4)
        self.enc3 = ResidualBlock(base_ch*4, base_ch*8)

        ch = [base_ch, base_ch*2, base_ch*4, base_ch*8]

        self.nodes = nn.ModuleDict()
        for j in range(2,5):
            for i in range(4-j+1):
                in_ch_node = ch[i]*(j-1) + ch[i+1]
                self.nodes[f"X{i}_{j}"] = ResidualBlock(in_ch_node, ch[i])

        self.up3 = nn.ConvTranspose2d(ch[3], ch[2], 2, stride=2)
        self.dec3 = ConvBlock(ch[2], ch[2])

        self.up2 = nn.ConvTranspose2d(ch[2], ch[1], 2, stride=2)
        self.dec2 = ConvBlock(ch[1], ch[1])

        self.up1 = nn.ConvTranspose2d(ch[1], ch[0], 2, stride=2)
        self.dec1 = ConvBlock(ch[0], ch[0])

    def forward(self, x):

        X0_1 = self.enc0(x)
        X1_1 = self.enc1(self.pool(X0_1))
        X2_1 = self.enc2(self.pool(X1_1))
        X3_1 = self.enc3(self.pool(X2_1))

        nodes = {"X0_1":X0_1,"X1_1":X1_1,"X2_1":X2_1,"X3_1":X3_1}

        for j in range(2,5):
            for i in range(4-j+1):
                horizontal = [nodes[f"X{i}_{k}"] for k in range(1,j)]
                lower = nodes[f"X{i+1}_{j-1}"]
                up = F.interpolate(lower, size=horizontal[0].shape[2:],
                                   mode="bilinear", align_corners=False)
                concat = torch.cat(horizontal+[up], dim=1)
                nodes[f"X{i}_{j}"] = self.nodes[f"X{i}_{j}"](concat)

        d3 = self.up3(X3_1) + nodes["X2_2"]
        d3 = self.dec3(d3)

        d2 = self.up2(d3) + nodes["X1_3"]
        d2 = self.dec2(d2)

        d1 = self.up1(d2) + nodes["X0_4"]
        d1 = self.dec1(d1)

        return d1  

class AttentionConcat(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch//8),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch//8, in_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        b,c,_,_ = x.shape
        w = self.avg_pool(x).view(b,c)
        w = self.fc(w).view(b,c,1,1)
        return x * w

class ResUNetPlusStrict3Branch(nn.Module):
    def __init__(self, base_ch=64, n_classes=3):
        super().__init__()
        
        self.branch_t1ce = StrictBranch(1, base_ch)
        self.branch_flair_t2 = StrictBranch(1, base_ch)
        self.branch_t1 = StrictBranch(1, base_ch)

        total_channels = base_ch * 3

        self.attention = AttentionConcat(total_channels)

        self.final = nn.Conv2d(total_channels, n_classes, 1)

    def forward(self, x):
        t1ce = x[:,0:1]
        flair_t2 = x[:,1:2]
        t1 = x[:,2:3]

        f1 = self.branch_t1ce(t1ce)
        f2 = self.branch_flair_t2(flair_t2)
        f3 = self.branch_t1(t1)

        fused = torch.cat([f1,f2,f3], dim=1)
        fused = self.attention(fused)

        return self.final(fused)