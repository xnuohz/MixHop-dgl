#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse

def parameter_parser():
    """
    MixHop Model Parameters
    """
    parser = argparse.ArgumentParser(description='MixHop GCN')

    # data source params
    parser.add_argument('--dataset', type=str, default='Cora', help='Name of dataset.')
    # model params
    parser.add_argument('--epochs', type=int, default=2000, help='Traning epochs.')
    parser.add_argument('--early-stopping', type=int, default=200, help='Early stopping.')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate.')
    parser.add_argument('--lamb', type=float, default=5e-2, help='L2 reg.')
    parser.add_argument('--gclayers', nargs='+', type=int, help='GC Layer dim.')
    parser.add_argument('--p', nargs='+', type=int, help='Power list of adj matrix.')

    parser.set_defaults(gclayers=[60, 60])
    parser.set_defaults(p=[0, 1, 2])

    return parser.parse_args()
