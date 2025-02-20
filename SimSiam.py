import torch
import torch.nn as nn
import torch.nn.functional as F



def Distance(p, z, version='original'):  # negative cosine similarity
    if version == 'original':
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimSiam(nn.Module):
    def __init__(self, backbone, size):
        super().__init__()
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)
        self.predictor = prediction_MLP()
        self.avg = nn.Linear(size*size, 1)
        self.avg2 = nn.Linear(size*size, 1)
        self.avg3 = nn.Linear(size*size, 1)


    def forward(self, x1, x2, x3):
        f1, f, h = self.backbone, self.projector, self.predictor
        l1 = f1(x1)
        l2 = f1(x2)
        l3 = f1(x3)
        # 提取类别标签的输出,因为在cat时将类别标签放在最前面
        D1, C1, H1, W1 = l1.size()
        l1 = l1.view(D1, C1, H1*W1)
        D1, C1, H1, W1 = l2.size()
        l2 = l2.view(D1, C1, H1 * W1)
        D1, C1, H1, W1 = l3.size()
        l3 = l3.view(D1, C1, H1 * W1)
        s_1 = self.avg(l1).squeeze()
        s_2 = self.avg2(l2).squeeze()
        s_3 = self.avg3(l3).squeeze()
        z1, z2, z3 = f(s_1), f(s_2), f(s_3)
        p1, p2, p3 = h(z1), h(z2), h(z3)
        # linear1 = l1
        # linear2 = l2
        # linear3 = l3
        # maeloss = (maeloss1 + maeloss2 + maeloss3)/3
        L = (Distance(p2, z1) + Distance(p3, z1) + Distance(p1, z2) + Distance(p1, z3)) / 4
        # L = (D(p1, z2) + D(p1, z3)) / 6 + (D(p2, z1) + D(p2, z3)) / 6 + (D(p3, z1) + D(p3, z2)) / 6

        return L


