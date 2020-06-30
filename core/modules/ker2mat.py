import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import toeplitz
from PIL import Image, ImageOps


class Kernel2MatrixConvertor:
    def __init__(self,im_shape, kers = [np.array([[1,0,-1],[2,0,-2],[1,0,-1]]),     #Dx
                                        np.array([[1,2,1],[0,0,0],[-1,-2,-1]])]):   #Dy
        
        self.kers = kers
        self.im_H, self.im_W = im_shape
        self.ker_H, self.ker_W = kers[0].shape
        self.D = [self.generate_matrix(ker) for ker in kers]

    def generate_matrix(self,ker):
        """
        Generates a matrix which performs 2D convolution between input I and filter F by converting the F to a toeplitz matrix and multiply it
          with vectorizes version of I
          By : AliSaaalehi@gmail.com
        
        """
        # number of columns and rows of the filter
        ker_H, ker_W = ker.shape

        #  calculate the matrix dimensions
        output_H = self.im_H + ker_H - 1
        output_W = self.im_W + ker_W - 1

        # zero pad the filter
        ker_padded = np.pad(ker, ((output_H - ker_H, 0),
                                   (0, output_W - ker_W)),
                                'constant', constant_values=0)

        # use each row of the zero-padded F to creat a toeplitz matrix. 
        #  Number of columns in this matrices are same as numbe of columns of input signal
        toeplitz_list = []
        for i in range(ker_padded.shape[0]-1, -1, -1): # iterate from last row to the first row
            c = ker_padded[i, :] # i th row of the F 
            r = np.r_[c[0], np.zeros(self.im_W-1)] # first row for the toeplitz fuction should be defined otherwise
                                                                # the result is wrong
            toeplitz_m = toeplitz(c,r) # this function is in scipy.linalg library
            toeplitz_list.append(toeplitz_m)

            # doubly blocked toeplitz indices: 
        #  this matrix defines which toeplitz matrix from toeplitz_list goes to which part of the doubly blocked
        c = range(1, ker_padded.shape[0]+1)
        r = np.r_[c[0], np.zeros(self.im_H-1, dtype=int)]
        doubly_indices = toeplitz(c, r)

        ## creat doubly blocked matrix with zero values
        toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
        h = toeplitz_shape[0]*doubly_indices.shape[0]
        w = toeplitz_shape[1]*doubly_indices.shape[1]
        doubly_blocked_shape = [h, w]
        doubly_blocked = np.zeros(doubly_blocked_shape)

        # tile toeplitz matrices for each row in the doubly blocked matrix
        b_h, b_w = toeplitz_shape # hight and withs of each block
        for i in range(doubly_indices.shape[0]):
            for j in range(doubly_indices.shape[1]):
                start_i = i * b_h
                start_j = j * b_w
                end_i = start_i + b_h
                end_j = start_j + b_w
                doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]

        return torch.from_numpy(doubly_blocked).type(torch.float32)

    
    def matrix_to_vector(self, matrix):
        return torch.flip(matrix, dims=[0]).flatten()

    def vector_to_matrix(self, vector):
        output_h, output_w = [self.im_H + self.ker_H - 1, self.im_W + self.ker_W - 1]
        return torch.flip(vector.reshape(output_h, output_w), dims=[0])