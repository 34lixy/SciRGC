import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime

import torch


def get_config():
    parser = argparse.ArgumentParser()
    '''Base'''

    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--model_name', type=str, default='roberta',
                        choices=['bert', 'roberta'])
    parser.add_argument('--method_name', type=str, default='fnn',
                        choices=['gru', 'rnn', 'bilstm', 'lstm', 'fnn', 'textcnn', 'attention', 'lstm+textcnn',
                                 'lstm_textcnn_attention'])

    '''Optimization'''
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--num_epoch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    '''Environment'''
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backend', default=False, action='store_true')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--timestamp', type=int, default='{:.0f}{:03}'.format(time.time(), random.randint(0, 999)))
    parser.add_argument('--patience_step', type=int, default=5)
    parser.add_argument('--output', type=str, default='output')  # 添加输出目录参数
    
    args = parser.parse_args()
    args.device = torch.device(args.device)
    
    
    '''logger'''
    args.log_name = '{}_{}_{}.log'.format(args.model_name, args.method_name,
                                          datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
    if not os.path.exists('logs'):
        os.mkdir('logs')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))
    
    # 确保输出目录存在
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists('figs'):
        os.mkdir('figs')
    
    return args, logger
