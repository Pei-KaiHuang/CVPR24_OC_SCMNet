import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import numpy as np
import random

def softplus(x):
    return torch.nn.functional.softplus(x, beta=100)

 
class block(nn.Module):

    def __init__(self,begin):
        super(block,self).__init__()

        if begin==True:
            self.cnn1=nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3,stride=1,padding=1)
        else:
            self.cnn1=nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3,stride=1,padding=1)
        
        nn.init.xavier_normal_(self.cnn1.weight)
        self.bn1=nn.BatchNorm2d(128,track_running_stats=False)
        # self.non_linearity1=nn.CELU(alpha=1.0, inplace=False)
        self.non_linearity1 = nn.ReLU(inplace=False)
        self.cnn2=nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3,stride=1,padding=1)
        nn.init.xavier_normal_(self.cnn2.weight)
        self.bn2=nn.BatchNorm2d(196,track_running_stats=False)
        # self.non_linearity2=nn.CELU(alpha=1.0, inplace=False)
        self.non_linearity2 = nn.ReLU(inplace=False)
        
        self.cnn3=nn.Conv2d(in_channels=196, out_channels=128, kernel_size=3,stride=1,padding=1)
        nn.init.xavier_normal_(self.cnn3.weight)
        self.bn3=nn.BatchNorm2d(128,track_running_stats=False)
        # self.non_linearity3=nn.CELU(alpha=1.0, inplace=False)
        self.non_linearity3 = nn.ReLU(inplace=False)

        
        self.pool=nn.MaxPool2d(kernel_size=2)

    def forward(self,x):
        
        x=self.cnn1(x)
        x=self.bn1(x)
        x=self.non_linearity1(x)
        x=self.cnn2(x)
        x=self.bn2(x)
        x=self.non_linearity2(x)
        x=self.cnn3(x)
        x=self.bn3(x)
        x=self.non_linearity3(x)
        x=self.pool(x)
        return x


class Simple_FeatureExtractor(nn.Module):
    
    def __init__(self):
        super(Simple_FeatureExtractor, self).__init__()
        self.resize_32 = nn.Upsample(size=32, mode='nearest')
        self.resize_64 = nn.Upsample(size=64, mode='nearest')

        self.cnn0=nn.Conv2d(in_channels=3,out_channels=64, kernel_size=3,stride=1,padding=1)
        nn.init.xavier_normal_(self.cnn0.weight)
        self.bn0=nn.BatchNorm2d(64,track_running_stats=False)
        # self.non_linearity0=nn.CELU(alpha=1.0, inplace=False)
        self.non_linearity0 = nn.ReLU(inplace=False)

        self.block1=block(True)
        self.block2=block(False)
        self.block3=block(False)
        
    def forward(self,x):
        x=self.cnn0(x)
        x=self.bn0(x)
        x=self.non_linearity0(x)
        
        #Block1
        x=self.block1(x)
        X1=self.resize_64(x)
        
        #Block2
        x=self.block2(x)
        X2=x
        
        #Block3:
        x=self.block3(x)
        X3=self.resize_64(x)
        
        X=torch.cat((X1,X2,X3),1)
        #print(X.size()) #torch.Size([2, 384, 64, 64])
        return x



class SCMNet(nn.Module):  
    def __init__(self, output=2):
        super(SCMNet, self).__init__() 

        self.F = Simple_FeatureExtractor()
        
        self.E = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

        self.G = self.net = nn.Sequential(
            nn.Conv2d(129, 256, kernel_size=4, stride=2, padding=1), 
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Upsample to 32x32
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Upsample to 64x64
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), 
            )
        
    def forward(self, x,ms=None, update = "learn_FE"):
        if self.training:
            
            if update == "learn_FE":
                self.F.requires_grad = True
                self.G.requires_grad = False
                self.E.requires_grad = True 
            elif update == "learn_Gtr":
                self.F.requires_grad = False
                self.G.requires_grad = True
                self.E.requires_grad = False 

            
            live_feature = F.normalize(self.F(x)) 
            
            noise = torch.randn(live_feature.shape).to('cuda')
            noise_ms_p = torch.cat([noise, ms], dim=1)
            
            partial_spoof_z = self.G(noise_ms_p)


            live_map = self.E(live_feature) 
            m_p = self.E(partial_spoof_z)

            return live_feature, partial_spoof_z, live_map, m_p

        else:
            
            feature = F.normalize(self.F(x))# torch.Size([2, 128, 32, 32])
            spoof_cue = self.E(feature) # torch.Size([2, 1, 32, 32])
            return spoof_cue
        
if __name__ == "__main__":
    net = SCMNet().cuda()
    net(torch.randn(5, 3, 256, 256).cuda())