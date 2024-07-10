import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim)
        self.dim = dim
    def forward(self, x_in):
        # [bs, 4*C, H, W])
        pos = self.pos_emb(x_in)
        x_in = x_in + pos
        b, c, h, w = x_in.shape
        x = x_in.permute(0, 2, 3, 1).reshape(b,h*w,c)
        # b, hw, hd
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        # b, h, hw, d
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q_inp, k_inp, v_inp))
        # b, h, d, hw
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        # attn: b, h, d, d
        attn = (k @ q.transpose(-2, -1))
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        # x: b, h, d, hw
        x = attn @ v
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        # out: b, c, h, w
        out_c = self.proj(x).view(b, h, w, c).permute(0, 3, 1, 2)

        return out_c

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads)
        self.ff = FeedForward(dim=dim)
    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x
    
class Conv3x3(nn.Module):
    def __init__(self, inc, outc, stride):
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, 3, stride, 1)
        self.bn = nn.BatchNorm2d(outc)
        self.relu = nn.LeakyReLU(0.2)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    
class MUnet(nn.Module):
    def __init__(self, inc, outc, midc=16, stages=4):
        super().__init__()
        self.inc, self.stages = inc, stages
        self.inconvs = nn.ModuleList()
        for i in range(inc):
            self.inconvs.append(Conv3x3(1, midc, 1))

        self.encs = nn.ModuleList()
        for i in range(inc):
            enc = nn.ModuleList()
            stagec = midc
            for k in range(stages):
                enc.append(Conv3x3(stagec, stagec*2, 2))
                stagec *= 2
            self.encs.append(enc)
        # cross modality module
        self.fusion = nn.ModuleList()
        stagec = midc
        for i in range(stages):
            stagec = stagec * 2
            self.fusion.append(nn.Sequential(
                Transformer(dim=inc*stagec, dim_head=midc, heads=stagec//midc),
                Conv3x3(inc*stagec, stagec, 1)
            ))

        self.dec = nn.ModuleList()
        self.convouts = nn.ModuleList()
        for k in range(stages):
            self.dec.append(nn.ModuleList([
                nn.ConvTranspose2d(stagec, stagec//2, 2, 2),
                Conv3x3(stagec, stagec//2, 1)
            ]))
            stagec //= 2
            # output
            self.convouts.append(nn.Conv2d(stagec, outc, 1, 1, 0))
    def forward(self, x):
        xin = torch.chunk(x, self.inc, 1)

        enc_feas = []
        for i in range(self.inc):
            feas = []
            x = self.inconvs[i](xin[i])
            for layer in self.encs[i]:
                x = layer(x)
                feas.append(x)
            enc_feas.append(feas)

        fus_feas = []
        for k in range(self.stages):
            feas = [feas[k] for feas in enc_feas]
            fea = torch.cat(feas, 1)
            fea = self.fusion[k](fea)
            fus_feas.append(fea)

        outs, embs = [], []
        fea = fus_feas[-1]
        for i, (up, merge) in enumerate(self.dec):
            fea = up(fea)
            if i < len(fus_feas)-1:
                fea = merge(torch.cat([fea, fus_feas[-2-i]], 1))
            out = self.convouts[i](fea)
            embs.append(embs)
            outs.append(out)
        if self.training:
            return outs
        else:
            return out