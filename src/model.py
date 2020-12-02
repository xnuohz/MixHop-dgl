#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import MixHopConv, ListModule
from tqdm import trange

class MixHopNetwork(nn.Module):
    """
        MixHop: Higher-Order Graph Convlutional Architecture via Sparsified Neighborhood Mixing.
    """
    def __init__(self, args, n_features, n_classes):
        super(MixHopNetwork, self).__init__()
        self.args = args
        self.n_features = n_features
        self.n_classes = n_classes
        self.calculate_layer_sizes()
        self.setup_layer_structure()

    def calculate_layer_sizes(self):
        self.args.gclayers = [self.n_features] + self.args.gclayers

        self.n_segs = int(self.args.gclayers[-1] / self.n_classes)
        if self.args.gclayers[-1] % self.n_classes != 0:
            print('Wasted columns: {} out of {}'.format(self.args.gclayers[-1] % self.n_classes, self.args.gclayers[-1]))

        self.q = torch.tensor(np.random.rand(self.n_segs), requires_grad=True)
        self.q = nn.Softmax(dim=0)(self.q)


    def setup_layer_structure(self):
        self.gc_layers = [
            MixHopConv(in_feats, out_feats, self.args.p, bias=True, activation=torch.tanh)
            for in_feats, out_feats in zip(self.args.gclayers[:-1], self.args.gclayers[1:])
        ]
        self.gc_layers = ListModule(*self.gc_layers)

    def calculate_loss(self):
        tot_weight_l2_loss = 0
        for gc_layer in self.gc_layers:
            tot_weight_l2_loss += gc_layer.calculate_loss()
        tot_weight_l2_loss += torch.norm(self.q)
        return self.args.lamb * tot_weight_l2_loss

    def forward(self, graph, feat):
        for gc_layer in self.gc_layers:
            feat = gc_layer(graph, feat)
        
        output = 0
        for k in range(self.n_segs):
            segment = feat[:, k * self.n_classes : (k + 1) * self.n_classes]
            output = segment * self.q[k] + output
        
        return feat

class Trainer:
    def __init__(self, args, graph):
        self.args = args
        self.graph = graph
        # data split
        self.train_mask = graph.ndata['train_mask']
        self.val_mask = graph.ndata['val_mask']
        self.test_mask = graph.ndata['test_mask']
        # data statistics
        self.n_nodes = graph.num_nodes()
        self.n_features = graph.ndata['feat'].shape[1]
        self.n_classes = max(graph.ndata['label']).item() + 1
        # model definition
        self.model = MixHopNetwork(args, self.n_features, self.n_classes)

    def fit(self):
        acc = 0
        no_improvement = 0
        epochs = trange(self.args.epochs, desc='Accuracy')
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.model.train()
        for _ in epochs:
            p = self.model(self.graph, self.graph.ndata['feat'])
            loss = F.cross_entropy(p[self.train_mask], self.graph.ndata['label'][self.train_mask])
            loss += self.model.calculate_loss()

            self.opt.zero_grad()
            loss.backward(retain_graph=True)
            self.opt.step()

            val_acc = self.predict(self.val_mask)
            epochs.set_description("Val Accuracy: {:04f}".format(val_acc))
            
            if val_acc < acc:
                no_improvement += 1
                if no_improvement == self.args.early_stopping:
                    epochs.close()
                    break
            else:
                no_improvement = 0
                acc = val_acc
        
        test_acc = self.predict(self.test_mask)
        print('\nTest Accuracy: {:04f}'.format(test_acc))

    def predict(self, node_mask):
        self.model.eval()
        p = self.model(self.graph, self.graph.ndata['feat']).argmax(dim=1)
        correct = p[node_mask].eq(self.graph.ndata['label'][node_mask]).sum().item()
        return correct / sum(node_mask)
