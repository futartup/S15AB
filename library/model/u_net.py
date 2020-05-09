import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthPrediction(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(DepthPrediction, self).__init__()
        