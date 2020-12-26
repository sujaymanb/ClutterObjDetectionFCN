import torch
import torch.nn as nn

class SuctionNet(nn.Module):
    def __init__(self, h, w, ch=32):
        super(SuctionNet, self).__init__()
        self.rgb_trunk = nn.Sequential(nn.Conv2d(3, ch, 5, 2),
                                        nn.ReLU(True),
                                        nn.Conv2d(ch,ch, 5, 2),
                                        nn.ReLU(True),
                                        nn.Conv2d(ch,ch, 5, 2),
                                        nn.ReLU(True),
                                        nn.Conv2d(ch,ch, 5, 1),
                                        nn.ReLU(True))

        self.depth_trunk = nn.Sequential(nn.Conv2d(1, ch, 5, 2),
                                        nn.ReLU(True),
                                        nn.Conv2d(ch,ch, 5, 2),
                                        nn.ReLU(True),
                                        nn.Conv2d(ch,ch, 5, 2),
                                        nn.ReLU(True),
                                        nn.Conv2d(ch,ch, 5, 1),
                                        nn.ReLU(True))

        self.head = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                    nn.Conv2d(ch*2,ch*2, 3, 1),
                                    nn.ReLU(True),
                                    nn.UpsamplingBilinear2d(scale_factor=2),
                                    nn.Conv2d(ch*2,1, 3, 1))

        self.h = h
        self.w = w


    def forward(self, rgb, depth):
        x1 = self.rgb_trunk(rgb)
        x2 = self.depth_trunk(depth)
        x = torch.cat([x1,x2],dim=1)
        out = self.head(x)
        out = nn.Upsample(size=(self.h,self.w), mode="bilinear").forward(out)

        return out



class SuctionNetRGB(nn.Module):
    def __init__(self, h, w):
        super(SuctionNetRGB, self).__init__()
        self.rgb_trunk = nn.Sequential(nn.Conv2d(3, 32, 5, 2),
                                        nn.ReLU(True),
                                        nn.Conv2d(32,32, 5, 2),
                                        nn.ReLU(True),
                                        nn.Conv2d(32,32, 5, 2),
                                        nn.ReLU(True),
                                        nn.Conv2d(32,32, 5, 1),
                                        nn.ReLU(True))
        self.head = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                    nn.Conv2d(32,32, 3, 1),
                                    nn.ReLU(True),
                                    nn.UpsamplingBilinear2d(scale_factor=2),
                                    nn.Conv2d(32,1, 3, 1))

        self.h = h
        self.w = w

    def forward(self, rgb):
        x = self.rgb_trunk(rgb)
        out = self.head(x)
        out = nn.Upsample(size=(self.h,self.w), mode="bilinear").forward(out)

        return out