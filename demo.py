import sys
sys.path.append('core')

import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

import datasets
from utils import flow_viz
from raft import RAFT

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = 'cuda'

def pad8(img):
    """pad image such that dimensions are divisible by 8"""
    ht, wd = img.shape[2:]
    pad_ht = (((ht // 8) + 1) * 8 - ht) % 8
    pad_wd = (((wd // 8) + 1) * 8 - wd) % 8
    pad_ht1 = [pad_ht//2, pad_ht-pad_ht//2]
    pad_wd1 = [pad_wd//2, pad_wd-pad_wd//2]

    img = F.pad(img, pad_wd1 + pad_ht1, mode='replicate')
    return img

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return pad8(img[None]).to(DEVICE)


def display(image1, image2, flow, flow_gt, name):
    image1 = image1.permute(1, 2, 0).cpu().numpy() / 255.0
    image2 = image2.permute(1, 2, 0).cpu().numpy() / 255.0

    flow = flow.permute(1, 2, 0).cpu().numpy()
    flow_image = flow_viz.flow_to_image(flow)

    flow_gt = flow_gt.permute(1, 2, 0).cpu().numpy()
    flow_gt_image = flow_viz.flow_to_image(flow_gt)

    im2save = np.concatenate([image1, image2, flow_image, flow_gt_image], axis = 1)
    error = np.sum(np.abs(flow - flow_gt)**2, axis=2)**0.5
    Image.fromarray((im2save*255).astype(np.uint8)).save(name)
    plt.imshow(error, cmap='gray'), plt.colorbar(), plt.savefig(name.replace(".png","_error.png"),bbox_inches='tight')



    #flow_image = cv2.resize(flow_image, (image1.shape[1], image1.shape[0]))


    #cv2.imshow('image1', image1[..., ::-1])
    #cv2.imshow('image2', image2[..., ::-1])
    #cv2.imshow('flow', flow_image[..., ::-1])
    #cv2.waitKey()
    #Image.fromarray((image1*255).astype(np.uint8)).save("./out/"+name+"_1.png")
    #Image.fromarray((image2*255).astype(np.uint8)).save("./out/"+name+"_2.png")
    #Image.fromarray((flow_image*255).astype(np.uint8)).save("./out/"+name+"_flo.png")




def demo(args):
    model = RAFT(args)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model))

    model.to(DEVICE)
    model.eval()

    with torch.no_grad():

        # sintel images
        image1 = load_image('images/sintel_0.png')
        image2 = load_image('images/sintel_1.png')

        flow_predictions = model(image1, image2, iters=args.iters, upsample=False)
        display(image1[0], image2[0], flow_predictions[-1][0], 'sintel')

        # kitti images
        image1 = load_image('images/kitti_0.png')
        image2 = load_image('images/kitti_1.png')

        flow_predictions = model(image1, image2, iters=16)    
        display(image1[0], image2[0], flow_predictions[-1][0], 'kitti')

        # davis images
        #image1 = load_image('images/davis_0.jpg')
        #image2 = load_image('images/davis_1.jpg')

        #flow_predictions = model(image1, image2, iters=16)    
        #display(image1[0], image2[0], flow_predictions[-1][0], 'davis')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--iters', type=int, default=12)

    args = parser.parse_args()
    demo(args)