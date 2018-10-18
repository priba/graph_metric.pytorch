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
from utils import load_checkpoint, graph_cuda, graph_to_sparse, graph_cat, knn_accuracy, mean_average_precision
from models import models

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"

def test(data_loader, gallery_loader, net, cuda, distance):
    batch_time = LogMetric.AverageMeter()
    acc = LogMetric.AverageMeter()
    meanap = LogMetric.AverageMeter()

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
            g = graph_to_sparse(g)
            if cuda:
                g = graph_cuda(g)
                target = target.cuda()

            # Output
            g_out  = net(g)

            target_gallery.append(target)
            g_gallery.append(g_out)

        target_gallery = torch.cat(target_gallery)
        g_gallery = graph_cat(g_gallery)

        target_query = []
        for i, (g, target) in enumerate(data_loader):
            g = graph_to_sparse(g)
            # Prepare input data
            if cuda:
                g = graph_cuda(g)
                target = target.cuda()
        
            # Output
            g_out  = net(g)
            d = distance(g_out, g_gallery, mode='retrieval')
            
            dist_matrix.append(d)
            target_query.append(target)

        dist_matrix = torch.stack(dist_matrix)
        target_query = torch.cat(target_query)
    
        # K-NN classifier
        acc.update(knn_accuracy(dist_matrix, target_gallery, target_query, k=5))
        
        # mAP retrieval
        meanap.update(mean_average_precision(dist_matrix, target_gallery, target_query))
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
    train_loader, valid_loader, test_loader, gallery_loader, in_size = load_data(args.dataset, args.data_path, triplet=args.triplet, batch_size=args.batch_size, prefetch=args.prefetch)
    
    print('Create model')
    net = models.GNN(in_size, args.out_size, nlayers=args.nlayers, hid=args.hidden) 

    print('Check CUDA')
    if args.cuda and args.ngpu > 1:
        print('\t* Data Parallel **NOT TESTED**')
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.cuda:
        print('\t* CUDA')
        net = net.cuda()
        
    start_epoch = 0
    best_acc = 0
    early_stop_counter = 0
    if args.load is not None:
        print('Loading model')
        checkpoint = load_checkpoint(args.load)
        net.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']

        print('Loaded model at epoch {epoch} and acc {acc}%'.format(epoch=checkpoint['epoch'],acc=checkpoint['best_acc']))

    print('***Test***')
    losses, acc, embedding = test(test_loader, net, args.cuda, criterion, evaluation, plot_data=True)

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

