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
from sklearn import metrics

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"

def test(data_triplet_loader, nets, cuda, data_pair_loader=None):
    batch_time = LogMetric.AverageMeter()
    acc = LogMetric.AverageMeter()
    auc = LogMetric.AverageMeter()

    net, distance = nets

    # switch to test mode
    net.eval()
    distance.eval()

    end = time.time()

    dist_matrix = []
    start = time.time()
    with torch.no_grad():
        total, correct = 0,0
        for j, (g1, g2, g3, target) in enumerate(data_triplet_loader):
            if cuda:
                g1.to(torch.device('cuda'))
                g2.to(torch.device('cuda'))
                g3.to(torch.device('cuda'))

            # Output
            g1 = net(g1)
            g2 = net(g2)
            g3 = net(g3)

            d_pos = distance(g1, g2, mode='pairs')
            d_neg = distance(g1, g3, mode='pairs')
            total += d_pos.shape[0]
            correct += (d_pos < d_neg).float().sum()

        acc.update(correct/total)

        if data_pair_loader is not None:
            distances, labels = [], []
            for j, (g1, g2, _, target) in enumerate(data_pair_loader):
                if cuda:
                    g1.to(torch.device('cuda'))
                    g2.to(torch.device('cuda'))

                # Output
                g1 = net(g1)
                g2 = net(g2)

                d = distance(g1, g2, mode='pairs')
                distances.append(d)
                labels.append(target)
            similarity = -torch.cat(distances, 0)
            similarity = (similarity-similarity.min()) / (similarity.max() - similarity.min() + 1e-8)
            labels = torch.cat(labels, 0)
            auc.update(metrics.roc_auc_score(labels.cpu(), similarity.cpu()))

        # mAP retrieval
    batch_time.update(time.time()-start)
    print('* Test Acc {acc.avg:.5f}; AUC {auc.avg: .5f} Time x Test {b_time.avg:.3f}'
            .format(acc=acc, auc=auc, b_time=batch_time))
    return acc, auc


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

