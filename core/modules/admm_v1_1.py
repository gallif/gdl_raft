import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class ADMMSolverBlock(nn.Module):
    def __init__(self,mask,rho,lamb,eta,learn_lamb=False,learn_eta=False,T=1):
        super(ADMMSolverBlock, self).__init__()
        # params
        self.T = T
        self.use_mask = mask
        # variables
        self.beta = None
        self.Q = None
        # blocks
        self.CalcGrad = SpatialGradients()
        self.SoftThresh = SoftThresholding(rho,lamb,learn_lamb)
        self.UpdateMul = MultiplierUpdate(eta,learn_eta)
        
        if self.use_mask:
            self.mask_gen = MaskGenerator()

    def forward(self, F, image):
        #generate mask
        if self.use_mask:
            mask = self.mask_gen(image)
        else:
            mask = None
        
        # initialize 
        C = self.CalcGrad(F)
        beta = torch.zeros_like(C)

        for t in range(self.T):
            Q = self.SoftThresh(C,beta,mask)
            beta = self.UpdateMul(Q,C,beta)

        return Q, C, beta
    
class SoftThresholding(nn.Module):
    def __init__(self,rho,lamb,learn_lamb=False):
        super(SoftThresholding, self).__init__()
        if learn_lamb:
            self.lamb = nn.Parameter(torch.rand(1,requires_grad=True))
        else:
            self.lamb = torch.tensor(lamb,requires_grad=False)
        self.rho = rho
    
    def forward(self,C, beta, mask):
        # apply mask
        if mask is not None:
            th = mask * torch.ones_like(C) * self.lamb.cuda() / self.rho
        else:
            th = torch.ones_like(C) * self.lamb.cuda() / self.rho

        Q = torch.zeros_like(C)
        valid = (C - beta).abs() >= th
        Q[valid] = C[valid] - beta[valid] - th[valid] * torch.sign(C[valid] - beta[valid]) 
        
        return Q

class MultiplierUpdate(nn.Module):
    def __init__(self, eta, learn_eta=False):
        super(MultiplierUpdate,self).__init__()
        if learn_eta:
            self.eta = nn.Parameter(torch.rand(1,requires_grad=True))
        else:
            self.eta = torch.tensor(eta,requires_grad=False)

    def forward(self, Q, C, beta):
        beta = beta + self.eta.cuda() * (Q - C)
        
        return beta

class MaskGenerator(nn.Module):
    def __init__(self):
        super(MaskGenerator,self).__init__()
        self.CalcGrad = SpatialGradients()

    def forward(self, image):
        image = (image + 1) / 2 # shift (-1,1) to (0,1)
        image = F.interpolate(image, scale_factor=1/8, mode='bilinear')
        # use gradient of source image
        im_gray = (image * torch.tensor([[[[0.2989]],[[0.5870]],[[0.1140]]]], dtype= torch.float).cuda()).sum(dim=1,keepdim=True)
        grads = self.CalcGrad(im_gray)
        mask = torch.exp(-torch.sum(grads**2, dim=1, keepdim=True).sqrt()) 
        return mask

class SpatialGradients(nn.Module):
    def __init__(self,  f_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                        f_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]):
        super(SpatialGradients,self).__init__()
        Dx = torch.tensor(f_x, dtype = torch.float, requires_grad = False).view(1,1,3,3).cuda()
        Dy = torch.tensor(f_y, dtype = torch.float, requires_grad = False).view(1,1,3,3).cuda()
        self.D = torch.cat((Dx, Dy), dim = 0)
    
    def forward(self,image):
        # apply filter over each channel seperately
        im_ch = torch.split(image, 1, dim = 1)
        grad_ch = [F.conv2d(ch, self.D.cuda(), padding = 1) for ch in im_ch]
        #Du = F.conv2d(u, D, padding = 1)
        #Dv = F.conv2d(v, D, padding = 1)
        return torch.cat(grad_ch, dim=1)