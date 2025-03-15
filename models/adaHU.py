import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.utils import optional_import

rearrange, _ = optional_import ("einops", name="rearrange")

class SoftClamp(nn.Module):

    def _init_(self, num_group: int = 2):
        super()._init_()
        self.num_group = num_group
        self.upper_bound = torch. tensor ([600., 1600.], requires_grad=True)
        self.lower_bound = torch. tensor ([-1000., 0.], requires_grad=True)
        self.upper_bound = nn.Parameter (self.upper_bound)
        self.lower_bound = nn.Parameter (self.lower_bound)
        self.register_parameter('upper_bound', self.upper_bound) 
        self.register_parameter('lower_bound', self.lower_bound)

    def forward (self, x):
        x = x. unsqueeze(1)
        u, l = self.get_clip_range()
        u = u.view(1, self.num_group, 1, 1, 1, 1)
        l = l.view(1, self.num_group, 1, 1, 1, 1)
        x = -(F.softplus(u - F.softplus(x - l) - l) - u)
        return x


    def get_clip_range(self):
        return self.upper_bound, self.lower_bound   


class GateClip(nn.Module):

    def _init_(self, in_channel: int, num_group: int = 2):
    
        super()._init_()
        
        self.num_group = num_group
        self.conv1 = nn.Conv3d(in_channel, 32, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv2 = nn. Conv3d(32, 16, kernel_size=3, stride=1, padding=0, bias=False)
        self.avgpool = nn.AdaptiveAvgPoo13d ((1, 1, 1))
        self.fc = nn.Linear(16, num_group)
        self.softclamp = SoftClamp (num_group)

    def forward (self, x):
        # b, c, d, h, w = x. shape
        b, _, _, _, _ = x.shape
        z = self.conv1(x)
        z = F.relu(z)
        z = self. conv2(z)
        z = F.relu(z)
        z = self.avgpool (z)
        z = z.view(z.size(0), -1)
        z = self.fc(z)
        x = torch.softmax(z.view(b, self.num_group, 1, 1, 1, 1), dim=1)*self.softclamp(x)
        return x.sum (1)
        
