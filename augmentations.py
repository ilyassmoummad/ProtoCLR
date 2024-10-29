import torch
import numpy as np
from torch import nn
from torchaudio import transforms as T


def random_time_shift(spec, Tshift):
    deltat = int(np.random.uniform(low=0.0, high=Tshift))
    if deltat == 0:
        return spec
    return torch.roll(spec, shifts=deltat, dims=-1)

class TimeShift(nn.Module):
    def __init__(self, Tshift):
        super().__init__()
        self.Tshift = Tshift

    def forward(self, spec):
        return random_time_shift(spec, self.Tshift)

def mix_random(x, min_coef=0.6):
    alpha = np.random.uniform(min_coef, 1.0, 1)[0]
    indices = torch.randperm(x.shape[0])
    return alpha * x + (1. - alpha) * x[indices,...]

class MixRandom(torch.nn.Module):
    def __init__(self, min_coef):
        super().__init__()
        self.min_coef = min_coef

    def forward(self, x):
        return mix_random(x, self.min_coef)
    
class SpecAugment(torch.nn.Module):
    def __init__(self, freq_mask=10, time_mask=30, freq_stripes=3, time_stripes=5, p=1.0, iid_masks=True):
        super().__init__()
        self.p = p
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.freq_stripes = freq_stripes
        self.time_stripes = time_stripes   
        self.specaugment = nn.Sequential(
            *[T.FrequencyMasking(freq_mask_param=self.freq_mask, iid_masks=iid_masks) for _ in range(self.freq_stripes)], 
            *[T.TimeMasking(time_mask_param=self.time_mask, iid_masks=iid_masks) for _ in range(self.time_stripes)],
            )
            
    def forward(self, audio):
        if self.p > torch.randn(1):
            return self.specaugment(audio)
        else:
            return audio