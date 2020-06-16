import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from ker2mat import Kernel2MatrixConvertor

class ADMMSolverBlock(nn.Module):
    def __init__(self,shape,rho,lamb,eta):
        super(ADMMSolverBlock, self).__init__()

        self.k2m = Kernel2MatrixConvertor(shape)
        self.Q = ReconstructionBlock(rho)
        self.C = ConvolutionBlock()
        self.Z = SoftThresholding(rho,lamb)
        self.M = MultiplierUpdate(eta)
    
    def forward(self,F,Z,Beta):
        D = self.k2m.D

        # Reconstruction
        Q_hat = self.Q(D,F,Z,Beta)
        # Convolution
        C_hat = [self.C(Q_hat, D_l, Beta_l) for D_l, Beta_l in zip(D,Beta)]
        # Soft Thresholding
        Z_hat = [self.Z(C_l) for C_l in C_hat]
        # Multipliers Update
        Beta_hat = [self.M(Q_hat, D_l, Z_l, Beta_l) for D_l, Z_l, Beta_l in zip(D,Z,Beta)]

        #for D_l, Beta_l in zip(D,Beta):
        #    C_l = self.C(Q_hat, D_l, Beta_l)
        #    Z_l = self.Z(C_l)
        #    Beta_l = self.M(Q_hat, D_l, Z_l, Beta_l)

        return Q_hat, (Beta_hat, Z_hat)


class ReconstructionBlock(nn.Module):
    def __init__(self,rho):
        super(ReconstructionBlock, self).__init__()
        self.rho = rho
    
    def forward(self,D,F,Z,Beta):
        rho = self.rho

        DtD = [Dl.T @ Dl for Dl in D]
        Dt_ZmBeta = [Dl.T @ (Zl - Beta_l) for Dl,Zl,Beta_l in zip(D,Z,Beta)]
        Q_hat = torch.inverse(torch.eye(F.shape[0]) + rho * torch.sum(DtD)) @ (F + rho * torch.sum(Dt_ZmBeta))

        return Q_hat

class ConvolutionBlock(nn.Module):
    def __init__(self):
        super(ConvolutionBlock, self).__init__()

    def forward(self,Q,D_l,Beta_l):
        C_l = D_l @ Q + Beta_l
        
        return C_l


class SoftThresholding(nn.Module):
    def __init__(self,rho,lamb):
        super(SoftThresholding, self).__init__()
        self.th = lamb / rho
    
    def forward(self,C_l):
        Z_l = torch.zeros_like(C_l)
        valid = C_l >= self.th
        Z_l[valid] = C_l[valid] - self.th * torch.sign(C_l[valid]) 
        
        return Z_l

class MultiplierUpdate(nn.Module):
    def __init__(self, eta):
        super(MultiplierUpdate,self).__init__()
        self.eta = eta

    def forward(self, Q, D_l, Z_l, Beta_l):
        Beta_l = Beta_l + self.eta * (D_l @ Q - Z_l)
        
        return Beta_l
