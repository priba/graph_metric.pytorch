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
from models import models, distance
from data.load_data import load_data
from loss.contrastive import ContrastiveLoss, TripletLoss
import dgl

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"

def test(data_loader, gallery_loader, nets, cuda):
    batch_time = LogMetric.AverageMeter()
    acc = LogMetric.AverageMeter()
    meanap = LogMetric.AverageMeter()

    net, distance = nets

    # switch to test mode
    net.eval()
    distance.eval()

    end = time.time()

    dist_matrix = []
    start = time.time()
    with torch.no_grad():
        g_gallery = []
        target_gallery = []
        for j, (g, target) in enumerate(gallery_loader):
            if cuda:
                g.ndata['h'] = g.ndata['h'].cuda()

            # Output
            g = net(g)

            target_gallery.append(target)
            g_gallery.append(g)

        target_gallery = np.array(np.concatenate(target_gallery))
        g_gallery = dgl.batch(g_gallery)

        target_query = []
        for i, (g, target) in enumerate(data_loader):
            # Prepare input data
            if cuda:
                g.ndata['h'] = g.ndata['h'].cuda()

            # Output
            g  = net(g)
            d = distance(g, g_gallery, mode='retrieval')

            dist_matrix.append(d)
            target_query.append(target)

        dist_matrix = torch.stack(dist_matrix)
        target_query = np.array(np.concatenate(target_query))

#        str_sim = (np.expand_dims(target_query, axis=1) == np.expand_dims(target_gallery, axis=0)) * 1
#        dist_sorted, ind_sorted = dist_matrix.sort()
#        sim_sorted = torch.gather(torch.from_numpy(str_sim).to(dist_matrix.device), dim=1, index=ind_sorted)

        # K-NN classifier
        acc.update(knn_accuracy(dist_matrix, target_gallery, target_query, k=5, dataset=data_loader.dataset.dataset))

        # mAP retrieval
        meanap.update(mean_average_precision(dist_matrix, target_gallery, target_query, dataset=data_loader.dataset.dataset))
    batch_time.update(time.time()-start)
    print('* Test Acc {acc.avg:.3f}; mAP {meanap.avg: .3f}; Time x Test {b_time.avg:.3f}'
            .format(acc=acc, meanap=meanap, b_time=batch_time))
    return acc, meanap


def main():
    print('Loss & Optimizer')
    if args.loss=='triplet':
        args.triplet=True
        criterion = TripletLoss(margin=args.margin, swap=args.swap)
    elif args.loss=='triplet_distance':
        args.triplet=True
        criterion = TripletLoss(margin=args.margin, swap=args.swap, dist=True)
    else:
        args.triplet=False
        criterion = ContrastiveLoss(margin=args.margin)

    print('Prepare data')
    train_loader, valid_loader, valid_gallery_loader, test_loader, test_gallery_loader, in_size = load_data(args.dataset, args.data_path, triplet=args.triplet, batch_size=args.batch_size, prefetch=args.prefetch)

    print('Create model')
    net = models.GNN(in_size, args.out_size, nlayers=args.nlayers, hid=args.hidden, J=args.pow)
    distNet = distance.SoftHd()

    print('Check CUDA')
    if args.cuda and args.ngpu > 1:
        print('\t* Data Parallel **NOT TESTED**')
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.cuda:
        print('\t* CUDA')
        net, distNet = net.cuda(), distNet.cuda()
        criterion = criterion.cuda()

    start_epoch = 0
    best_map = 0
    early_stop_counter = 0
    if args.load is not None:
        print('Loading model')
        checkpoint = load_checkpoint(args.load)
        net.load_state_dict(checkpoint['state_dict'])
        distNet.load_state_dict(checkpoint['state_dict_dist'])
        start_epoch = checkpoint['epoch']
        best_map = checkpoint['best_map']
        print('Loaded model at epoch {epoch} and mAP {meanap}%'.format(epoch=checkpoint['epoch'],meanap=checkpoint['best_map']))

    print('***Test***')
    test(test_loader, test_gallery_loader, [net, distNet], args.cuda)

if __name__ == '__main__':
    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))

    # Check cuda & Set random seed
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Check Test and Load
    if args.load is None:
        raise Exception('Cannot test without loading a model.')

    main()

