#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import dgl

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from param_parser import parameter_parser
from model import Trainer
from texttable import Texttable

def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([['Parameter', 'value']])
    t.add_rows([[k.replace('-', ' ').capitalize(), args[k]] for k in keys])
    print(t.draw())


def get_graph(name, raw_dir='./data'):
    """
        name: ['Cora', 'Citeseer', 'Pubmed']
    """ 
    if name == 'Cora':
        dataset = CoraGraphDataset(raw_dir=raw_dir)
    elif name == 'Citeseer':
        dataset = CiteseerGraphDataset(raw_dir=raw_dir)
    elif name == 'Pubmed':
        dataset = PubmedGraphDataset(raw_dir=raw_dir)
    else:
        raise ValueError('Dataset name is invalid.')
    
    g = dataset[0]
    return dgl.add_self_loop(g)


def main():
    args = parameter_parser()
    tab_printer(args)
    graph = get_graph(args.dataset)

    trainer = Trainer(args, graph)
    trainer.fit()

if __name__ == "__main__":
    main()
