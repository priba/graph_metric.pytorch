 # -*- coding: utf-8 -*-

"""
    Parse input arguments
"""

import argparse

__author__ = 'Pau Riba'
__email__ = 'priba@cvc.uab.cat'


class Options():

    def __init__(self):
        # MODEL SETTINGS
        parser = argparse.ArgumentParser(description='Train a Metric Learning Graph Neural Network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # Positional arguments
        parser.add_argument('dataset', type=str, choices=['iam', 'histograph-gw', 'histograph-ak'], help='Dataset.')
        parser.add_argument('data_path', type=str, help='Dataset root path.')
        # Model parameters
        parser.add_argument('--model', type=str, default='GAT', help='Model to use GAT or GRU', choices=['GAT', 'GRU'])
        parser.add_argument('--set_partition', type=str, help='Dataset set partition (Only histograph-gw otherwise it is ignored).', default='cv1')
        parser.add_argument('--batch_size', '-bz', type=int, default=64, help='Batch Size.')
        parser.add_argument('--out_size', type=int, default=32, help='Node Embedding size.')
        parser.add_argument('--hidden', type=int, default=32, help='Hidden size.')
        parser.add_argument('--dropout', type=float, default=0.3, help='Dropout.')
        parser.add_argument('--loss', type=str, default='siamese', choices=['siamese', 'triplet', 'triplet_distance'], help='Loss used for training.')
        parser.add_argument('--swap', action='store_true', help='Swap in the triplet loss.')
        parser.add_argument('--margin', type=float, default=10, help='Margin in the loss function.')
        # Optimization options
        parser.add_argument('--prefetch', type=int, default=4, help='Number of workers to load data.')
        parser.add_argument('--epochs', '-e', type=int, default=1000, help='Number of epochs to train.')
        parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='The Learning Rate.')
        parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
        parser.add_argument('--schedule', type=int, nargs='+', default=[],
                            help='Decrease learning rate at these epochs.')
        parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        # Checkpoints
        parser.add_argument('--save', '-s', type=str, default=None, help='Folder to save checkpoints.')
        parser.add_argument('--load', '-l', type=str, default=None, help='Checkpoint path to resume / test.')
        parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
        parser.add_argument('--early_stop', '-es', type=int, default=50, help='Early stopping epochs.')
        # Acceleration
        parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU, 1 = CUDA, 1 < DataParallel')
        # i/o
        parser.add_argument('--log', type=str, default=None, help='Log folder.')
        parser.add_argument('--log-interval', type=int, default=256, metavar='N',
                            help='How many batches to wait before logging training status')
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()

