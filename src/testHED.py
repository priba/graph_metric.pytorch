# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Graph classification
"""

# Python modules
import torch
import glob
import numpy as np
import time
import os
import argparse

# Own modules
from options import Options
from Logger import LogMetric
from utils import load_checkpoint, knn_accuracy, mean_average_precision, mean_average_precision_Bo
from models import models, realdistance
from data.load_data import load_data
import dgl

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"

def test(data_loader, gallery_loader, distance, cuda):
    batch_time = LogMetric.AverageMeter()
    acc = LogMetric.AverageMeter()
    meanap = LogMetric.AverageMeter()
    meanapbo = LogMetric.AverageMeter()

    end = time.time()
    distance.eval()

    dist_matrix = []
    start = time.time()
    with torch.no_grad():
        g_gallery = []
        target_gallery = []
        for j, (g, target) in enumerate(gallery_loader):
            if cuda:
                g.to(torch.device('cuda'))
                g.gdata['std'] = g.gdata['std'].cuda()

            target_gallery.append(target)
            g_gallery.append(g)

        target_gallery = np.array(np.concatenate(target_gallery))
        gdata = list(map(lambda g: g.gdata['std'], g_gallery))
        g_gallery = dgl.batch(g_gallery)
        g_gallery.gdata = {'std': torch.cat(gdata)}

        target_query = []
        for i, (g, target) in enumerate(data_loader):
            # Prepare input data
            if cuda:
                g.to(torch.device('cuda'))
                g.gdata['std'] = g.gdata['std'].cuda()

            # Output
            d = distance(g, g_gallery, mode='retrieval')

            dist_matrix.append(d)
            target_query.append(target)

        dist_matrix = torch.stack(dist_matrix)
        target_query = np.array(np.concatenate(target_query))


        # K-NN classifier
        acc.update(knn_accuracy(dist_matrix, target_gallery, target_query, k=5, dataset=data_loader.dataset.dataset))

        # mAP retrieval
        meanap.update(mean_average_precision(dist_matrix, target_gallery, target_query))
        meanapbo.update(mean_average_precision_Bo(dist_matrix, target_gallery, target_query))
    batch_time.update(time.time()-start)
    print('* Test Acc {acc.avg:.3f}; mAP {meanap.avg: .3f}; mAP {meanapbo.avg: .3f}; Time x Test {b_time.avg:.3f}'
            .format(acc=acc, meanap=meanap, meanapbo=meanapbo, b_time=batch_time))
    return acc, meanap


def main():
    print('Prepare data')
    train_loader, valid_loader, valid_gallery_loader, test_loader, test_gallery_loader, in_size = load_data(args.dataset, args.data_path, batch_size=args.batch_size, prefetch=args.prefetch)

    distance = realdistance.HausdorffEditDistance(alpha=args.alpha, beta=args.beta, tau_n=args.tau_n, tau_e=args.tau_e)

    print('Check CUDA')
    if args.cuda and args.ngpu > 1:
        distance = torch.nn.DataParallel(distance, device_ids=list(range(args.ngpu)))

    if args.cuda:
        distance = distance.cuda()

    print('***Test***')
    test(test_loader, test_gallery_loader, distance, args.cuda)

if __name__ == '__main__':
    # Parse options
    parser = argparse.ArgumentParser(description='Train a Metric Learning Graph Neural Network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('dataset', type=str, choices=['iam', 'histograph-gw', 'histograph-ak'], help='Dataset.')
    parser.add_argument('data_path', type=str, help='Dataset root path.')
    # Model parameters
    parser.add_argument('--set_partition', type=str, help='Dataset set partition (Only histograph-gw otherwise it is ignored).', default='cv1')
    parser.add_argument('--batch_size', '-bz', type=int, default=64, help='Batch Size.')
    parser.add_argument('--tau_n', type=float, default=4, help='Batch Size.')
    parser.add_argument('--tau_e', type=float, default=16, help='Batch Size.')
    parser.add_argument('--alpha', type=float, default=0.5, help='Batch Size.')
    parser.add_argument('--beta', type=float, default=0.1, help='Batch Size.')
    # Optimization options
    parser.add_argument('--prefetch', type=int, default=4, help='Number of workers to load data.')
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU, 1 = CUDA, 1 < DataParallel')
    # i/o
    args = parser.parse_args()

    # Check cuda & Set random seed
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()

    main()

