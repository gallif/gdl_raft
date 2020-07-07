import sys
sys.path.append('core')

from PIL import Image
import cv2
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

import core.datasets as datasets
from core.utils import flow_viz
from core.raft import RAFT
from demo import display

SAVE_FREQ = 50

def validate_chairs(args, model, iters=12):
    """ Evaluate trained model on Flying Chairs """
    model.eval()
    #pad = 0

    val_dataset = datasets.FlyingChairs(args, do_augument=False)
    print('Evaluating over FlyingChairs...')    

    epe_list = []
    for i in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, _ = val_dataset[i]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        #image1 = F.pad(image1, [0, 0, pad, pad], mode='replicate')
        #image2 = F.pad(image2, [0, 0, pad, pad], mode='replicate')
        with torch.no_grad():
            flow_predictions = model.module(image1, image2, iters=iters)
            flow_pr = flow_predictions[-1][0,:,:]
        epe = torch.sum((flow_pr - flow_gt.cuda())**2, dim=0)
        epe = torch.sqrt(epe).mean()
        epe_list.append(epe.item())
    print("Validation (%s) EPE: %f" % ('Chairs', np.mean(epe_list)))



def validate_sintel(args, model, iters=50):
    """ Evaluate trained model on Sintel(train) clean + final passes """
    model.eval()
    pad = 2

    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(args, do_augument=False, dstype=dstype)
        
        epe_list = []
        for i in tqdm(range(len(val_dataset))):
            image1, image2, flow_gt, _ = val_dataset[i]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            image1 = F.pad(image1, [0, 0, pad, pad], mode='replicate')
            image2 = F.pad(image2, [0, 0, pad, pad], mode='replicate')

            with torch.no_grad():
                flow_predictions = model.module(image1, image2, iters=iters)
                flow_pr = flow_predictions[-1][0,:,pad:-pad]

            epe = torch.sum((flow_pr - flow_gt.cuda())**2, dim=0)
            epe = torch.sqrt(epe).mean()
            epe_list.append(epe.item())

            if args.save_images and i % SAVE_FREQ == 0:
                display(image1[0,:,pad:-pad], image2[0,:,pad:-pad], flow_pr, flow_gt, os.path.join(args.log_dir, dstype + "_{}.png".format(i)))




        print("Validation (%s) EPE: %f" % (dstype, np.mean(epe_list)))


def validate_kitti(args, model, iters=32):
    """ Evaluate trained model on KITTI (train) """

    model.eval()
    val_dataset = datasets.KITTI(args, do_augument=False, is_val=True, do_pad=True)

    with torch.no_grad():
        epe_list, out_list = [], []
        for i in range(len(val_dataset)):
            image1, image2, flow_gt, valid_gt = val_dataset[i]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            flow_gt = flow_gt.cuda()
            valid_gt = valid_gt.cuda()

            flow_predictions = model.module(image1, image2, iters=iters)
            flow_pr = flow_predictions[-1][0]

            epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()
            mag = torch.sum(flow_gt**2, dim=0).sqrt()
            
            epe = epe.view(-1)
            mag = mag.view(-1)
            val = valid_gt.view(-1) >= 0.5

            out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
            epe_list.append(epe[val].mean().item())
            out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)


    print("Validation KITTI: %f, %f" % (np.mean(epe_list), 100*np.mean(out_list)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='name your results')
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--sintel_iters', type=int, default=50)
    parser.add_argument('--kitti_iters', type=int, default=32)
    parser.add_argument('--dataset', help='dataset used for eval')
    parser.add_argument('--data_dir', help='path to dataset')
    parser.add_argument('--cuda_devices', default="0,1", help="choose which GPUs are available")
    parser.add_argument('--save_images', action='store_true', help='dump images during evaluation')
    parser.add_argument('--batch_size', type=int, default=1)
    
    parser.add_argument('--admm_solver', action='store_true', help='apply admm block')
    parser.add_argument('--admm_mask',action='store_true', help='apply mask within admm block')
    parser.add_argument('--admm_lamb', type=float, default=0.4)
    parser.add_argument('--admm_rho', type=float, default=0.4)
    parser.add_argument('--admm_eta', type=float, default=0.4)

    args = parser.parse_args()
    args.log_dir = os.path.join(os.getcwd(), 'out', args.name)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    if args.dataset == 'sintel':
        args.image_size = [440, 1024]

    print('----Script Args----')
    for arg in vars(args):
        print('{} : {}'.format(arg, getattr(args, arg)))

    model = RAFT(args)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model))
    print('Loaded model for eval : ' + args.model)

    if (not os.path.isdir(args.log_dir) and args.save_images) is True:
        os.mkdir(args.log_dir)
        print("Images will be saved to " + args.log_dir)


    model.to('cuda')
    model.eval()
    if args.dataset == 'chairs':
        validate_chairs(args, model)
    elif args.dataset == 'sintel':
        validate_sintel(args, model, args.sintel_iters)
    elif args.dataset == 'kitti':
        validate_kitti(args, model, args.kitti_iters)    
