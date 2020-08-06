import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from core.modules.ker2mat import Kernel2MatrixConvertor

class ADMMInitiator(nn.Module):
    def __init__(self,admm_mask,shape,rho):
        super(ADMMInitiator,self).__init__()
        
        Dx, Dy = Kernel2MatrixConvertor(shape[:2]).D
        self.Dx = nn.Parameter(Dx, requires_grad = False)
        self.Dy = nn.Parameter(Dy, requires_grad = False)
        self.rho = rho
        self.admm_mask = admm_mask
        if admm_mask:
            self.masks_gen = MaskGenerator()

    def forward(self, image):
        # Extract D matrices
        D = [self.Dx, self.Dy]
        # Generate M matrices
        if self.admm_mask:
            masks = self.masks_gen(image)
            # reshape and create masked matrices M
            masks_rsh = [torch.flip(m, dims=[2]).flatten(start_dim=2) for m in masks]
            M = [m * d[None] for m,d in zip(masks_rsh,D)]
        else:
            M = D
        
        # Generate B matrix (inverse)
        MtM = [torch.matmul(Ml.permute(0,2,1), Ml) for Ml in M]
        R = torch.eye(MtM[0].shape[1]).cuda() + self.rho * sum(MtM)
        B = self.schur_comp_inverse(R)
        #Bt = torch.inverse(R)

        return (M,B),masks

    def schur_comp_inverse(self,R,MAX=512):
        n = R.shape[2]; p = n // 2
        A = R[:,:p,:p]; B = R[:,:p,p:]; C = R[:,p:,:p]; D = R[:,p:,p:]
        if p < MAX:
            D_inv = torch.inverse(D)
            RmD_inv = torch.inverse(A - B @ D_inv @ C)
        else:
            D_inv = self.schur_comp_inverse(D)
            RmD_inv = self.schur_comp_inverse(A - B @ D_inv @ C)

        return torch.cat(   [torch.cat([RmD_inv, -RmD_inv @ B @ D_inv], dim=2),
                            torch.cat([-D_inv @ C @ RmD_inv, D_inv + D_inv @ C @ RmD_inv @ B @ D_inv], dim=2)],
                            dim=1)
            

class ADMMSolverBlock(nn.Module):
    def __init__(self,shape,rho,lamb,eta,T=1):
        super(ADMMSolverBlock, self).__init__()
        
        self.T = T
        self.u_solver = ADMMSolverBlockPerChannel(shape,rho,lamb,eta)
        self.v_solver = ADMMSolverBlockPerChannel(shape,rho,lamb,eta)

    def forward(self, F, Matrices):
        M, B = Matrices
        #reshape flow and seperate into channels
        F_rsh = torch.flip(F, dims=[2]).flatten(start_dim=2).permute(0,2,1)
        F_u, F_v = torch.split(F_rsh, 1, dim=2)

        #treat each channel independently
        self.u_solver.reset()
        self.v_solver.reset()
        for t in range(self.T+1):
            Q_u, _ = self.u_solver(F_u, M, B)
            Q_v, _ = self.v_solver(F_v, M, B)

        #merge 2 channels and restore original shape
        out_H, out_W = F.shape[2], F.shape[3]

        Q_u = torch.flip(Q_u.reshape(-1, out_H, out_W), dims=[1])[None].permute(1,0,2,3)
        Q_v = torch.flip(Q_v.reshape(-1, out_H, out_W), dims=[1])[None].permute(1,0,2,3)

        Q = torch.cat([Q_u, Q_v], dim=1)

        return Q


class ADMMSolverBlockPerChannel(nn.Module):
    def __init__(self,shape,rho,lamb,eta,pad=2):
        super(ADMMSolverBlockPerChannel, self).__init__()

        # initialize blocks
        self.reconstruct = ReconstructionBlock(rho)
        self.convolve = ConvolutionBlock()
        self.apply_threshold = SoftThresholding(rho,lamb)
        self.update = MultiplierUpdate(eta)

        # initialize with zeros
        self.shape = shape
        self.pad = pad
        self.Beta_x = torch.zeros((shape[2],(shape[0]+pad)*(shape[1]+pad),1))
        self.Beta_y = torch.zeros((shape[2],(shape[0]+pad)*(shape[1]+pad),1))
        self.Z_x = torch.zeros((shape[2],(shape[0]+pad)*(shape[1]+pad),1))
        self.Z_y = torch.zeros((shape[2],(shape[0]+pad)*(shape[1]+pad),1))

    def forward(self,F, M, B):
        # Extract hidden variables
        Z = [self.Z_x.cuda(), self.Z_y.cuda()]
        Beta = [self.Beta_x.cuda(), self.Beta_y.cuda()]

        # Reconstruction
        Q_hat = self.reconstruct(F,M,B,Z,Beta)
        # Convolution
        C_hat = [self.convolve(Q_hat, M_l, Beta_l) for M_l, Beta_l in zip(M,Beta)]
        # Soft Thresholding
        Z_hat = [self.apply_threshold(C_l) for C_l in C_hat]
        # Multipliers Update
        Beta_hat = [self.update(Q_hat, M_l, Z_l, Beta_l) for M_l, Z_l, Beta_l in zip(M,Z,Beta)]

        # Update hidden variables
        self.Z_x, self.Z_y = Z_hat
        self.Beta_x, self.Beta_y = Beta_hat

        return Q_hat, (Beta_hat, Z_hat)
    
    def reset(self):
        self.Beta_x.zero_()
        self.Beta_y.zero_()
        self.Z_x.zero_()
        self.Z_y.zero_()
        
        return


class ReconstructionBlock(nn.Module):
    def __init__(self,rho):
        super(ReconstructionBlock, self).__init__()
        self.rho = rho
        
    def forward(self,F,M,B,Z,Beta):
        rho = self.rho

        Mt_ZmBeta = [torch.matmul(Ml.permute(0,2,1), Zl - Beta_l) for Ml,Zl,Beta_l in zip(M,Z,Beta)]
        Q_hat = torch.matmul(B, F + rho * sum(Mt_ZmBeta))

        return Q_hat


class ConvolutionBlock(nn.Module):
    def __init__(self):
        super(ConvolutionBlock, self).__init__()

    def forward(self,Q,M_l,Beta_l):
        C_l = torch.matmul(M_l, Q) + Beta_l
        
        return C_l


class SoftThresholding(nn.Module):
    def __init__(self,rho,lamb):
        super(SoftThresholding, self).__init__()
        self.lamb = lamb
        self.rho = rho
    
    def forward(self,C_l):
        th = self.lamb / self.rho

        Z_l = torch.zeros_like(C_l)
        valid = C_l.abs() >= th
        Z_l[valid] = C_l[valid] - th * torch.sign(C_l[valid]) 
        
        return Z_l


class MultiplierUpdate(nn.Module):
    def __init__(self, eta):
        super(MultiplierUpdate,self).__init__()
        self.eta = eta

    def forward(self, Q, M_l, Z_l, Beta_l):
        Beta_l = Beta_l + self.eta * (torch.matmul(M_l, Q) - Z_l)
        
        return Beta_l


class MaskGenerator(nn.Module):
    def __init__(self):
        super(MaskGenerator,self).__init__()
        self.sobel = Sobel()
        self.ddx_encoder = MaskEncoder()
        self.ddy_encoder = MaskEncoder()
    
    def rgb2gray(self, im_rgb):
        #im_rgb = (im_rgb + 1) / 2 # shift (-1,1) to (0,1)
        im_gray = (im_rgb * torch.tensor([[[[0.2989]],[[0.5870]],[[0.1140]]]], dtype= torch.float).cuda()).sum(dim=1,keepdim=True)
        return im_gray

    def forward(self, image):
        im_grads = self.sobel(self.rgb2gray(image))
        im_grads = torch.split(im_grads, 1, dim = 1)
        encoders = [self.ddx_encoder, self.ddy_encoder]
        masks = [enc(grad) for enc, grad in zip(encoders, im_grads)]

        return masks


class MaskEncoder(nn.Module):
    def __init__(self, cin=1):
        super(MaskEncoder,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = cin, out_channels = 8, kernel_size = 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.out = nn.Conv2d(in_channels = 32, out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
        self.relu = nn.ReLU(inplace = True)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))    #spatial dims are //2
        x = self.relu(self.conv2(x))    #spatial dims are //4
        x = self.relu(self.conv3(x))    #spatial dims are //8
        x = self.sig(self.out(x))       #cout = 1, vals are in [0,1]
        
        return x


class Sobel(nn.Module):
    def __init__(self,  f_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                        f_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]):
        super(Sobel, self).__init__()
        Dx = torch.tensor(f_x, dtype = torch.float, requires_grad = False).view(1,1,3,3)
        Dy = torch.tensor(f_y, dtype = torch.float, requires_grad = False).view(1,1,3,3)
        self.D = nn.Parameter(torch.cat((Dx, Dy), dim = 0), requires_grad = False)
    
    def forward(self,image):
        # apply filter over each channel seperately
        im_ch = torch.split(image, 1, dim = 1)
        grad_ch = [F.conv2d(ch, self.D, padding = 1) for ch in im_ch]
        return torch.cat(grad_ch, dim=1)