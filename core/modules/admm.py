import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy.linalg import circulant

class ADMMSolverBlock(nn.Module):
    def __init__(self):
        super(ADMMSolverBlock, self).__init__()
        #self.Q = ReconstructionBlock()
        #self.Z = SoftThresholding()
    
    def DiffMatrices(self,H,W):
        # Generate differentiation matrix operators of size HWxHW
        dx = np.zeros((H*W,1)); dx[0] = -1; dx[H] = 1
        Dx = torch.from_numpy(circulant(dx).T)

        dy = np.zeros((H*W,1)); dy[0] = -1; dy[1] = 1
        Dy = torch.from_numpy(circulant(dy).T)

        return (Dx,Dy)
    
    def forward(self,F,Z,beta):

        return


class ReconstructionBlock(nn.Module):
    def __init__(self):
        super(ReconstructionBlock, self).__init__()
    
    def forward(self,D,F,Z,beta):
        return

class SoftThresholding(nn.Module):
    def __init__(self):
        super(SoftThresholding, self).__init__()
    
    def forward(self,C,th):
        return


a = ADMMSolverBlock()
a.DiffMatrices(10,10)