# -*- coding: utf-8 -*- 
from __future__ import print_function, division

"""
Graph classification
"""

# Python modules
import torch
from torch.autograd.variable import Variable
import glob
import numpy as np
import time
import os
import matplotlib.pyplot as plt

# Own modules
from options import Options
from Logger import LogMetric
from utils import load_checkpoint, accuracy
from models import models

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

def test(data_loader, net, cuda, criterion, evaluation, plot_data=False):
    batch_time = LogMetric.AverageMeter()
    losses = LogMetric.AverageMeter()
    acc = LogMetric.AverageMeter()

    # switch to train mode
    net.eval()

    end = time.time()

    emb = None
    emb_pred = None
    emb_real = None
    with torch.no_grad():
        for i, (nodes, edges, target) in enumerate(data_loader):
            # Prepare input data
            if cuda:
                nodes, edges, target = nodes.cuda(), edges.cuda(), target.cuda()
        
            # Output
            out  = net(nodes, edges)
        
            loss = criterion(out, target)
            bacc = evaluation(out, target)

            # Save values
            losses.update(loss.data[0], int(nl[:,-1].sum()))
            acc.update(bacc[0].data[0], int(nl[:,-1].sum()))

            batch_time.update(time.time() - end)
            end = time.time()

    print('* Test Average Loss {loss.avg:.3f}; Avg Acc {acc.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
            .format(loss=losses, acc=acc, b_time=batch_time))
    return losses, acc


def main():
    print('Prepare data')
    train_loader, valid_loader, test_loader = load_data(args.data_path, batch_size=args.batch_size, num_workers=args.prefetch, pin_memory=True)
    
    print('Create model')
    net = None

    print('Loss & Optimizer')
    criterion = torch.nn.NLLLoss()

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

