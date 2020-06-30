import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from core.modules.ker2mat import Kernel2MatrixConvertor

class ADMMSolverBlock(nn.Module):
    def __init__(self,shape,rho,lamb,eta):
        super(ADMMSolverBlock, self).__init__()

        self.u_solver = ADMMSolverBlockPerChannel(shape,rho,lamb,eta)
        self.v_solver = ADMMSolverBlockPerChannel(shape,rho,lamb,eta)

    def forward(self, F):
        #reshape flow and seperate into channels
        F_u = torch.flip(F[:,0,:,:], dims=[1]).flatten(start_dim=1).T
        F_v = torch.flip(F[:,1,:,:], dims=[1]).flatten(start_dim=1).T

        #treat each channel independently
        Q_u, _ = self.u_solver(F_u)
        Q_v, _ = self.v_solver(F_v)

        #merge 2 channels
        out_H, out_W = F.shape[2], F.shape[3]

        Q_u = torch.flip(Q_u.reshape(out_H, out_W, -1), dims=[0])[None].permute(3,0,1,2)
        Q_v = torch.flip(Q_v.reshape(out_H, out_W, -1), dims=[0])[None].permute(3,0,1,2)

        Q = torch.cat([Q_u, Q_v], dim=1)

        return Q



class ADMMSolverBlockPerChannel(nn.Module):
    def __init__(self,shape,rho,lamb,eta):
        super(ADMMSolverBlockPerChannel, self).__init__()

        self.Q = ReconstructionBlock(rho)
        self.C = ConvolutionBlock()
        self.Z = SoftThresholding(rho,lamb)
        self.M = MultiplierUpdate(eta)

        # initialize with zeros
        padH, padW = 3-1, 3-1
        self.Beta_hat = [torch.zeros(((shape[0]+padH)*(shape[1]+padW),shape[2]))]*2
        self.Z_hat = [torch.zeros(((shape[0]+padH)*(shape[1]+padW),shape[2]))]*2

        # initialize filtering matrices
        self.D = [Dl for Dl in Kernel2MatrixConvertor(shape[:2]).D]

    
    def forward(self,F):
        D = [D_l.cuda() for D_l in self.D]
        Z = [Z_l.cuda() for Z_l in self.Z_hat]
        Beta = [Beta_l.cuda() for Beta_l in self.Beta_hat]

        # Reconstruction
        Q_hat = self.Q(D,F,Z,Beta)
        # Convolution
        C_hat = [self.C(Q_hat, D_l, Beta_l) for D_l, Beta_l in zip(D,Beta)]
        # Soft Thresholding
        Z_hat = [self.Z(C_l) for C_l in C_hat]
        # Multipliers Update
        Beta_hat = [self.M(Q_hat, D_l, Z_l, Beta_l) for D_l, Z_l, Beta_l in zip(D,Z,Beta)]

        # Update hidden variables
        self.Z_hat = Z_hat
        self.Beta_hat = Beta_hat

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
        Q_hat = torch.inverse(torch.eye(F.shape[0]).cuda() + rho * sum(DtD)) @ (F + rho * sum(Dt_ZmBeta))

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
