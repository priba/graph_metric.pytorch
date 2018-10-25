# -*- coding: utf-8 -*- 
from __future__ import print_function, division

"""
    Graph metric learning
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
from utils import save_checkpoint, load_checkpoint, graph_cuda, graph_to_sparse
from models import models, distance
from test import test
from data.load_data import load_data
from loss.contrastive import ContrastiveLoss, TripletLoss
__author__ = "Pau Riba"
__email__ = "priba@cvc.uab.cat"


def adjust_learning_rate(optimizer, epoch):
    """
        Updates the learning rate given an schedule and a gamma parameter.
    """
    if epoch in args.schedule:
        args.learning_rate *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate


def train(data_loader, nets, optimizer, cuda, criterion, epoch):
    batch_time = LogMetric.AverageMeter()
    losses = LogMetric.AverageMeter()

    net, distNet = nets
    # switch to train mode
    net.train()
    distNet.train()

    end = time.time()
    for i, (g1, g2, g3, target) in enumerate(data_loader):
        g1, g2 = graph_to_sparse(g1), graph_to_sparse(g2)
        if args.triplet:
            g3 = graph_to_sparse(g3)
        # Prepare input data
        if cuda:
            g1, g2 = graph_cuda(g1), graph_cuda(g2)
            if args.triplet:
                g3 = graph_cuda(g3)
            else:
                target = target.cuda()
        
        optimizer.zero_grad()
        
        # Output
        g1_out = net(g1)
        g2_out = net(g2)

        if args.triplet:
            g3_out = net(g3)
            loss = criterion(g1_out, g2_out, g3_out, distNet)
        else:
            loss = criterion(g1_out, g2_out, target, distNet)
        
        # Gradiensts and update
        loss.backward()
        optimizer.step()
        
        # Save values
        losses.update(loss.item(), g1[2].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i > 0 and i%args.log_interval == 0:
            print('Epoch: [{0}]({1}/{2}) Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
                    .format(epoch, i, len(data_loader), loss=losses, b_time=batch_time))
    print('Epoch: [{0}] Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
            .format(epoch, loss=losses, b_time=batch_time))
    return losses


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
    
    optimizer = torch.optim.SGD(list(net.parameters())+list(distNet.parameters()), args.learning_rate, momentum=args.momentum, weight_decay=args.decay, nesterov=True)

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

    if not args.test:
        print('***Train***')
        
        for epoch in range(start_epoch, args.epochs):
            # Update learning rate
            adjust_learning_rate(optimizer, epoch)

            loss_train = train(train_loader, [net, distNet], optimizer, args.cuda, criterion, epoch)
            acc_valid, map_valid = test(valid_loader, valid_gallery_loader, [net, distNet], args.cuda)
            
            # Early-Stop + Save model
            if map_valid.avg > best_map:
                best_map = map_valid.avg
                early_stop_counter = 0
                if args.save is not None:
                    save_checkpoint({'epoch': epoch + 1, 'state_dict': net.state_dict(), 'state_dict_dist': distNet.state_dict(), 'best_map': best_map}, directory=args.save, file_name='checkpoint')
            else:
                if early_stop_counter == args.early_stop:
                    break
                early_stop_counter += 1

            # Logger
            if args.log:
                # Scalars
                logger.add_scalar('loss_train', loss_train.avg)
                logger.add_scalar('acc_valid', acc_valid.avg)
                logger.add_scalar('map_valid', map_valid.avg)
                logger.add_scalar('learning_rate', args.learning_rate)
                logger.step()
        
        # Load Best model in case of save it
        if args.save is not None:
            print('Loading best  model')
            best_model_file = os.path.join(args.save, 'checkpoint.pth')
            checkpoint = load_checkpoint(best_model_file)
            net.load_state_dict(checkpoint['state_dict'])
            distNet.load_state_dict(checkpoint['state_dict_dist'])
            print('Best model at epoch {epoch} and acc {acc}%'.format(epoch=checkpoint['epoch'],acc=checkpoint['best_map']))

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
    if args.test and args.load is None:
        raise Exception('Cannot test without loading a model.')

    if not args.test and args.log is not None:
        print('Initialize logger')
        ind = len(glob.glob(args.log + '*_run-batchSize_{}'.format(args.batch_size)))
        log_dir = args.log + '{}_run-batchSize_{}/' \
                .format(ind, args.batch_size)
        args.save = args.save + '{}_run-batchSize_{}/' \
                .format(ind, args.batch_size)
        # Create logger
        print('Log dir:\t' + log_dir)
        logger = LogMetric.Logger(log_dir, force=True)

    main()

