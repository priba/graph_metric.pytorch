# -*- coding: utf-8 -*- 
from __future__ import print_function, division

"""
    Graph metric learning
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
from utils import save_checkpoint, load_checkpoint
from models import models
from test import test
from data.load_data import load_data

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


def train(data_loader, net, optimizer, cuda, criterion, epoch):
    batch_time = LogMetric.AverageMeter()
    losses = LogMetric.AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()
    for i, (g1, g2, g3, target) in enumerate(data_loader):
        import pdb; pdb.set_trace()
        # Prepare input data
        if cuda:
            nodes, edges, target = nodes.cuda(), edges.cuda(), target.cuda()
        
        optimizer.zero_grad()
        
        # Output
        out = net(nodes, edges)

        loss = criterion(out, target)
        
        # Gradiensts and update
        loss.backward()
        optimizer.step()
        
        # Save values
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: [{0}] Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
            .format(epoch, loss=losses, b_time=batch_time))
    return losses


def main():
    print('Prepare data')
    train_loader, valid_loader, test_loader = load_data(args.dataset, args.data_path, triplet=args.triplet, batch_size=args.batch_size)

    print('Create model')
    net = models.GNN(7, 5) 

    print('Loss & Optimizer')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay, nesterov=True)

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

    if not args.test:
        print('***Train***')
        
        for epoch in range(start_epoch, args.epochs):
            # Update learning rate
            adjust_learning_rate(optimizer, epoch)

            loss_train = train(train_loader, net, optimizer, args.cuda, criterion, epoch)
            loss_valid, acc_valid, _ = test(valid_loader, net, args.cuda, criterion, evaluation)
            
            # Early-Stop + Save model
            if acc_valid.avg > best_acc:
                best_acc = acc_valid.avg
                early_stop_counter = 0
                if args.save is not None:
                    save_checkpoint({'epoch': epoch + 1, 'state_dict': net.state_dict(), 'best_acc': best_acc}, directory=args.save, file_name='checkpoint')
            else:
                if early_stop_counter == args.early_stop:
                    break
                early_stop_counter += 1

            # Logger
            if args.log:
                # Scalars
                logger.add_scalar('loss_train', loss_train.avg)
                logger.add_scalar('loss_valid', loss_valid.avg)
                logger.add_scalar('acc_valid', acc_valid.avg)
                logger.add_scalar('learning_rate', args.learning_rate)
                logger.step()
        
        # Load Best model in case of save it
        if args.save is not None:
            print('Loading best  model')
            best_model_file = os.path.join(args.save, 'checkpoint.pth')
            checkpoint = load_checkpoint(best_model_file)
            net.load_state_dict(checkpoint['state_dict'])
            print('Best model at epoch {epoch} and acc {acc}%'.format(epoch=checkpoint['epoch'],acc=checkpoint['best_acc']))

    print('***Test***')
    test(test_loader, net, args.cuda, criterion, evaluation)

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
        log_dir = args.log + '{}_run-batchSize_{}/' \
                .format(len(glob.glob(args.log + '*_run-batchSize_{}'.format(args.batch_size))), args.batch_size)

        # Create logger
        print('Log dir:\t' + log_dir)
        logger = LogMetric.Logger(log_dir, force=True)

    main()

