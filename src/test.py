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

# Own modules
from options import Options
from Logger import LogMetric
from utils import load_checkpoint
from models import models

__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"

def test(data_loader, gallery_loader, net, cuda, distance):
    batch_time = LogMetric.AverageMeter()
    losses = LogMetric.AverageMeter()
    acc = LogMetric.AverageMeter()

    # switch to test mode
    net.eval()
    distance.eval()

    end = time.time()

    dist_matrix = []
    ind_i = []
    ind_j = []
    with torch.no_grad():
        for i, (g1, target1) in enumerate(data_loader):
            # Prepare input data
            if cuda:
                g1 = tuple((gi.cuda() for gi in g1) )
                target1 = target1.cuda()
        
            # Output
            g1_out  = net(g1)
            
            ind_j_aux = []
            dist_j = []
            for j, (g2, target2) in enumerate(gallery_loader):
                if cuda:
                    g2 = tuple((gi.cuda() for gi in g2) )
                    target2 = target2.cuda()

                # Output
                g2_out  = net(g2)

                # Batch size
                bz = target2.shape[0]
                g = (g1_out[0].repeat(bz,1))
                d = distance(g1_out, g2_out, mode='retrieval')
              
                dist_j.append(d)
                ind_j_aux.append(target2)

            dist_matrix.append(torch.cat(dist_i))
            ind_i.append(target1)
            ind_j.append(torch.cat(ind_j_aux))
        import pdb; pdb.set_trace()

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

