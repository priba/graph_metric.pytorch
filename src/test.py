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
from sklearn.metrics import average_precision_score
from joblib import Parallel, delayed
import multiprocessing

# Own modules
from options import Options
from Logger import LogMetric
from utils import load_checkpoint, graph_cuda, graph_to_sparse
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
        
        predict = knn_predict(dist_matrix, target_gallery)
        bacc = predict.eq(target_query).float().sum()
        acc.update(100.0*bacc/predict.size(0))
        
        # mAP retrieval
        nq = target_query.size(0)
        sim = 1./(1+dist_matrix)
        str_sim = target_query.unsqueeze(1) == target_gallery
        if sim.is_cuda:
            sim, str_sim = sim.cpu(), str_sim.cpu()
        sim, str_sim = sim.numpy(), str_sim.numpy()

        num_cores = min(multiprocessing.cpu_count(), 32)
        aps = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(str_sim[iq], sim[iq]) for iq in range(nq))

        meanap.update(np.mean(aps))
    batch_time.update(time.time()-start)
    print('* Test Acc {acc.avg:.3f}; mAP {meanap.avg: .3f}; Time x Test {b_time.avg:.3f}'
            .format(acc=acc, meanap=meanap, b_time=batch_time))
    return acc, meanap


def knn_predict(dist_matrix, target_gallery, k=5):
    ind = np.argsort(dist_matrix, axis=1)
    sort_target = target_gallery[ind]
    sort_target = sort_target[:,:k]
    
    # Counts
    counts = np.zeros(sort_target.shape)
    for i in range(k):
        counts[:,i] = (sort_target[:, i].unsqueeze(1) == sort_target).long().sum(1)

    predict_ind = counts.argmax(axis=1)
    predict = [sort_target[i, pi] for i, pi in enumerate(predict_ind)]
    return torch.stack(predict)

def graph_cat(g):
    nodes = []
    indices = []
    data = []
    g_size = []
    offset = 0
    for gi in g:
        nodes.append(gi[0])
        indices.append(gi[1]._indices()+offset)
        data.append(gi[1]._values())
        g_size.append(gi[2])
        offset = offset + gi[0].size(0)
    g_out = (
            torch.cat(nodes),
            torch.cat(indices, dim=1),
            torch.cat(data),
            torch.cat(g_size)
            )
    return graph_to_sparse(g_out)


def main():
    print('Prepare data')
    train_loader, valid_loader, test_loader, gallery_loader, in_size = load_data(args.dataset, args.data_path, triplet=args.triplet, batch_size=args.batch_size, prefetch=args.prefetch)
    
    print('Create model')
    net = models.GNN(in_size, args.out_size, nlayers=args.nlayers, hid=args.hidden) 

    print('Loss & Optimizer')
    print('Loss & Optimizer')
    if args.triplet:
        criterion = TripletLoss(margin=args.margin, swap=args.swap)
    else:
        criterion = ContrastiveLoss(margin=args.margin)

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

