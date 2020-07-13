import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from core.modules.ker2mat import Kernel2MatrixConvertor

class ADMMSolverBlock(nn.Module):
    def __init__(self,shape,mask,rho,lamb,eta,T=1):
        super(ADMMSolverBlock, self).__init__()
        
        self.T = T
        self.D = [Dl.cuda() for Dl in Kernel2MatrixConvertor(shape[:2]).D]
        self.u_solver = ADMMSolverBlockPerChannel(self.D,shape,rho,lamb,eta)
        self.v_solver = ADMMSolverBlockPerChannel(self.D,shape,rho,lamb,eta)
        
        self.use_mask = mask
        if self.use_mask:
            print('mask is enabled')
            self.mask_gen = MaskGenerator(shape,pad=1)

    def forward(self, F, image):
        #generate mask
        if self.use_mask:
            mask = self.mask_gen(image)[:,0,:,:].flatten(start_dim=1).T
        else:
            mask = None

        #reshape flow and seperate into channels
        F_u = torch.flip(F[:,0,:,:], dims=[1]).flatten(start_dim=1).T
        F_v = torch.flip(F[:,1,:,:], dims=[1]).flatten(start_dim=1).T

        #treat each channel independently
        self.u_solver.reset()
        self.v_solver.reset()
        for t in range(self.T):
            Q_u, _ = self.u_solver(F_u, mask)
            Q_v, _ = self.v_solver(F_v, mask)
            #update
            F_u,F_v = Q_u,Q_v

        #merge 2 channels
        out_H, out_W = F.shape[2], F.shape[3]

        Q_u = torch.flip(Q_u.reshape(out_H, out_W, -1), dims=[0])[None].permute(3,0,1,2)
        Q_v = torch.flip(Q_v.reshape(out_H, out_W, -1), dims=[0])[None].permute(3,0,1,2)

        Q = torch.cat([Q_u, Q_v], dim=1)

        return Q



class ADMMSolverBlockPerChannel(nn.Module):
    def __init__(self,D,shape,rho,lamb,eta,pad=2):
        super(ADMMSolverBlockPerChannel, self).__init__()
        
        # initialize filtering matrices
        self.D = D

        # initialize blocks
        self.Q = ReconstructionBlock(self.D,rho)
        self.C = ConvolutionBlock()
        self.Z = SoftThresholding(rho,lamb)
        self.M = MultiplierUpdate(eta)

        # initialize with zeros
        self.shape = shape
        self.pad = pad
        self.Beta_hat = None
        self.Z_hat = None

    def forward(self,F, mask=None):
        D = [D_l.cuda() for D_l in self.D]
        Z = [Z_l.cuda() for Z_l in self.Z_hat]
        Beta = [Beta_l.cuda() for Beta_l in self.Beta_hat]

        # Reconstruction
        Q_hat = self.Q(F,Z,Beta)
        # Convolution
        C_hat = [self.C(Q_hat, D_l, Beta_l) for D_l, Beta_l in zip(D,Beta)]
        # Soft Thresholding
        Z_hat = [self.Z(C_l, mask) for C_l in C_hat]
        # Multipliers Update
        Beta_hat = [self.M(Q_hat, D_l, Z_l, Beta_l) for D_l, Z_l, Beta_l in zip(D,Z,Beta)]

        # Update hidden variables
        self.Z_hat = Z_hat
        self.Beta_hat = Beta_hat

        return Q_hat, (Beta_hat, Z_hat)
    
    def reset(self):
        pad = self.pad
        shape = self.shape
        self.Beta_hat = [torch.zeros(((shape[0]+pad)*(shape[1]+pad),shape[2]))]*2
        self.Z_hat = [torch.zeros(((shape[0]+pad)*(shape[1]+pad),shape[2]))]*2
        
        return



class ReconstructionBlock(nn.Module):
    def __init__(self,D,rho):
        super(ReconstructionBlock, self).__init__()
        self.rho = rho
        self.D = D
        
        # calculate inverse matrix once
        DtD = [Dl.T @ Dl for Dl in self.D]
        self.B = torch.inverse(torch.eye(DtD[0].shape[0]).cuda() + rho * sum(DtD))

    def forward(self,F,Z,Beta):
        rho = self.rho

        Dt_ZmBeta = [Dl.cuda().T @ (Zl - Beta_l) for Dl,Zl,Beta_l in zip(self.D,Z,Beta)]
        Q_hat = self.B.cuda() @ (F + rho * sum(Dt_ZmBeta))

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
        self.lamb = lamb
        self.rho = rho
    
    def forward(self,C_l, mask):
        # apply mask
        if mask is not None:
            th = mask * self.lamb / self.rho
        else:
            th = torch.ones_like(C_l) * self.lamb / self.rho

        Z_l = torch.zeros_like(C_l)
        valid = C_l.abs() >= th
        Z_l[valid] = C_l[valid] - th[valid] * torch.sign(C_l[valid]) 
        
        return Z_l

class MultiplierUpdate(nn.Module):
    def __init__(self, eta):
        super(MultiplierUpdate,self).__init__()
        self.eta = eta

    def forward(self, Q, D_l, Z_l, Beta_l):
        Beta_l = Beta_l + self.eta * (D_l @ Q - Z_l)
        
        return Beta_l

class MaskGenerator(nn.Module):
    def __init__(self,shape,pad=0,learned_mask=False):
        super(MaskGenerator,self).__init__()
        self.learned_mask = learned_mask
        self.pad = pad
        if not learned_mask:
            Dx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype = torch.float).view(1,1,3,3)
            Dy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype = torch.float).view(1,1,3,3)
            self.D = torch.cat((Dx, Dy), dim = 0)

    def forward(self, image):
        image = (image + 1) / 2 # shift (-1,1) to (0,1)
        image = F.interpolate(image, scale_factor=1/8, mode='bilinear')
        #image = F.pad(image,[self.pad, self.pad, self.pad, self.pad])
        if not self.learned_mask:
            # use gradient of source image
            im_gray = (image * torch.tensor([[[[0.2989]],[[0.5870]],[[0.1140]]]], dtype= torch.float).cuda()).sum(dim=1,keepdim=True)
            grads = F.conv2d(im_gray, self.D.cuda(), padding = 1)
            mask = torch.exp(-torch.sum(grads**2, dim=1, keepdim=True).sqrt()) 
        return mask