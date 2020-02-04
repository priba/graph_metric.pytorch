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

# Own modules
from options import Options
from Logger import LogMetric
from utils import load_checkpoint, knn_accuracy, mean_average_precision
from models import models, realdistance
from data.load_data import load_data
import dgl

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"

def test(data_loader, gallery_loader, distance, cuda):
    batch_time = LogMetric.AverageMeter()
    acc = LogMetric.AverageMeter()
    meanap = LogMetric.AverageMeter()

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
    batch_time.update(time.time()-start)
    print('* Test Acc {acc.avg:.3f}; mAP {meanap.avg: .3f}; Time x Test {b_time.avg:.3f}'
            .format(acc=acc, meanap=meanap, b_time=batch_time))
    return acc, meanap


def main():
    print('Prepare data')
    train_loader, valid_loader, valid_gallery_loader, test_loader, test_gallery_loader, in_size = load_data(args.dataset, args.data_path, batch_size=args.batch_size, prefetch=args.prefetch)

    print('Create model')
    distance = realdistance.HausdorffEditDistance()

    print('Check CUDA')
    if args.cuda and args.ngpu > 1:
        print('\t* Data Parallel **NOT TESTED**')
        distance = torch.nn.DataParallel(distance, device_ids=list(range(args.ngpu)))

    if args.cuda:
        print('\t* CUDA')
        distance = distance.cuda()

    print('***Test***')
    test(test_loader, test_gallery_loader, distance, args.cuda)

if __name__ == '__main__':
    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))

    # Check cuda & Set random seed
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()

    main()

