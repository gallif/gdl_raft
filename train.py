from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from logger import TBLogger
from pathlib import Path
from torch.utils.data import DataLoader
from core.raft import RAFT
from evaluate import validate_sintel, validate_kitti
import core.datasets as datasets
from core.utils.flow_viz import flow_to_image

# exclude extremly large displacements
MAX_FLOW = 1000
SUM_FREQ = 100
VAL_FREQ = 5000


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sequence_loss(flow_preds, flow_gt, valid, sup_loss = 'l1', tv_weight = 0.0):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    valid = (valid >= 0.5) & (flow_gt.abs().sum(dim=1) < MAX_FLOW)

    for i in range(n_predictions):
        i_weight = 0.8**(n_predictions - i - 1)
        
        if sup_loss == 'l1':
            i_loss = (flow_preds[i] - flow_gt).abs()
        elif sup_loss == 'l2':
            i_loss = (flow_preds[i] - flow_gt)**2
        
        if tv_weight > 0.0:
            i_tv = tv_weight * total_variation(flow_preds[i])
        else:
            i_tv = 0.0

        flow_loss += i_weight * (valid[:, None] * (i_loss + i_tv)).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'loss': flow_loss.item(),
        'epe':  epe.mean().item(),
        '1px':  (epe < 1).float().mean().item(),
        '3px':  (epe < 3).float().mean().item(),
        '5px':  (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def total_variation(flow):
    Dx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype = torch.float, requires_grad = False).view(1,1,3,3).cuda()
    Dy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype = torch.float, requires_grad = False).view(1,1,3,3).cuda()
    D = torch.cat((Dx, Dy), dim = 0)

    u,v = torch.split(flow, 1, dim = 1)
    Du = F.conv2d(u, D, padding = 1)
    Dv = F.conv2d(v, D, padding = 1)
    
    return torch.cat((Du.sum(dim = 1, keepdim = True), Dv.sum(dim = 1, keepdim = True)), dim = 1)


def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    if args.dataset == 'chairs':
        train_dataset = datasets.FlyingChairs(args, root=args.data_dir, image_size=args.image_size)
    
    elif args.dataset == 'things':
        clean_dataset = datasets.SceneFlow(args, root=args.data_dir, image_size=args.image_size, dstype='frames_cleanpass')
        final_dataset = datasets.SceneFlow(args, root=args.data_dir, image_size=args.image_size, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.dataset == 'sintel':
        clean_dataset = datasets.MpiSintel(args, image_size=args.image_size, dstype='clean')
        final_dataset = datasets.MpiSintel(args, image_size=args.image_size, dstype='final')
        train_dataset = clean_dataset + final_dataset

    elif args.dataset == 'kitti':
        train_dataset = datasets.KITTI(args, image_size=args.image_size, is_val=False)

    gpuargs = {'num_workers': 4, 'drop_last' : True}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, **gpuargs)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps,
        pct_start=0.2, cycle_momentum=False, anneal_strategy='linear', final_div_factor=1.0)

    return optimizer, scheduler
    
class Logger:
    def __init__(self, model, scheduler, name):
        self.model = model
        self.scheduler = scheduler
        self.name = name
        self.total_steps = 0
        self.running_loss = {}

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        name_str = self.name + " : "
        
        # print the training status
        print(name_str + training_str + metrics_str)

        #for key in self.running_loss:
        #    self.running_loss[key] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        if self.total_steps % SUM_FREQ == 0:
            self.running_loss = {}

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()


def train(args):

    model = RAFT(args)
    model = nn.DataParallel(model)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt))

    model.cuda()
    model.train()
    
    if 'chairs' not in args.dataset:
        model.module.freeze_bn()

    train_loader = fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    logger = Logger(model, scheduler, args.name)
    tb_logger = TBLogger(args.log_dir)

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            optimizer.zero_grad()
            flow_predictions = model(image1, image2, iters=args.iters)
            
            loss, metrics = sequence_loss(flow_predictions, flow, valid, sup_loss=args.sup_loss, tv_weight = args.tv_weight)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            total_steps += 1

            logger.push(metrics)

            if total_steps % SUM_FREQ == SUM_FREQ-1:
                # Scalar Summaries
                # ============================================================
                tb_logger.scalar_summary('lr', optimizer.param_groups[0]['lr'], total_steps)
                for key, value in logger.running_loss.items():
                    tb_logger.scalar_summary(key, value, total_steps)

                # Image Summaries
                # ============================================================
                B = args.batch_size
                """
                vis_batch = []
                for b in range(B):
                    batch = [np.array(fl_pred[b].detach().squeeze(0).cpu()).transpose(1,2,0) for fl_pred in flow_predictions]
                    vis = batch + [flow[b].detach().cpu().numpy().transpose(1,2,0)]
                    vis = np.concatenate(list(map(flow_to_image, vis)), axis = 1)
                    vis_batch.append(vis)
                tb_logger.image_summary(f'flow', vis_batch, total_steps)
                """
                # Images vs. Pred vs. GT
                gt_list = [np.array(x) for x in np.array(flow.detach().cpu()).transpose(0,2,3,1).tolist()]
                pr_list = [np.array(x) for x in np.array(flow_predictions[-1].detach().cpu()).transpose(0,2,3,1).tolist()]
                gt_list = list(map(flow_to_image, gt_list))
                pr_list = list(map(flow_to_image, pr_list))
                tb_logger.image_summary('src & tgt, pred, gt', 
                    [np.concatenate([np.concatenate([i.squeeze(0), j.squeeze(0)], axis = 1), np.concatenate([k, l], axis = 1)], axis=0) 
                        for i,j,k,l in zip(   np.split(np.array(image1.data.cpu()).astype(np.uint8).transpose(0,2,3,1), B, axis = 0), 
                                            np.split(np.array(image2.data.cpu()).astype(np.uint8).transpose(0,2,3,1), B, axis = 0),
                                            gt_list,
                                            pr_list)
                    ], 
                    total_steps)

                # Error
                pred_batch = [np.array(x) for x in np.array(flow_predictions[-1].detach().cpu()).transpose(0,2,3,1).tolist()]
                gt_batch = [np.array(x) for x in np.array(flow.detach().cpu()).transpose(0,2,3,1).tolist()]
                err_batch = [(np.sum(np.abs(pr - gt)**2, axis=2,keepdims=True)**0.5).astype(np.uint8) for pr,gt in zip(pred_batch, gt_batch)]
                err_vis = [np.concatenate([gt, pr, np.tile(err,(1,1,3))], axis=0) for gt, pr, err in zip(gt_list, pr_list,err_batch )]
                tb_logger.image_summary(f'Error', err_vis, total_steps)

            if (total_steps % VAL_FREQ == VAL_FREQ-1 and args.save_checkpoints) is True:
                PATH = args.log_dir + '/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

            if total_steps == args.num_steps:
                should_keep_training = False
                break


    PATH = args.log_dir +'/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_devices', default="0,1", help="choose which GPUs are available")
    parser.add_argument('--name', default='bla', help="name your experiment")
    parser.add_argument('--dataset', help="which dataset to use for training") 
    parser.add_argument('--data_dir', help='path to dataset')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--save_checkpoints', default=True, help='save checkpoints during training')
    parser.add_argument('--log_dir', default = os.path.join(os.getcwd(), 'checkpoints', datetime.now().strftime('%Y%m%d-%H%M%S')))

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--sup_loss', help='supervised loss term', default='l1')
    parser.add_argument('--tv_weight', type=float, help='total variation term weight', default=0.0)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if (not os.path.isdir(args.log_dir) and args.save_checkpoints) is True:
        os.mkdir(args.log_dir)
        print("Checkpoints will be saved to " + args.log_dir)

    # scale learning rate and batch size by number of GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    num_gpus = torch.cuda.device_count()
    args.batch_size = args.batch_size * num_gpus
    args.lr = args.lr * num_gpus

    train(args)
