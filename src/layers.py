#!/usr/bin/env python3
# -*- coding:utf-8 -*-
""" Torch Module for MixHop GC layer """

import torch
import torch.nn as nn
import dgl.function as fn

class MixHopConv(nn.Module):
    def __init__(self, in_feats, out_feats, p=[0, 1, 2], bias=True, activation=None, dropout=0.5):
        super(MixHopConv, self).__init__()
        self.weights = {k: nn.Linear(in_feats, out_feats, bias=bias) for k in p}
        self._p = p
        self._activation = activation
        self.fc = nn.Linear(len(p) * out_feats, out_feats, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for _, v in self.weights.items():
            nn.init.xavier_uniform_(v.weight)
            if v.bias is not None:
                nn.init.zeros_(v.bias)
        
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def calculate_loss(self):
        weight_l2_loss = 0
        for _, v in self.weights.items():
            weight_l2_loss += torch.norm(v.weight)
        return weight_l2_loss

    def forward(self, graph, feat):
        with graph.local_scope():
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).to(feat.device).unsqueeze(1)
            max_j = max(self._p) + 1
            outputs = []
            for j in range(max_j):
                feat = feat * norm
                graph.ndata['h'] = feat
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                feat = graph.ndata.pop('h')
                feat = feat * norm

                if j in self._p:
                    output = self.weights[j](feat)
                    if self._activation:
                        output = self._activation(output)
                    outputs.append(output)
            return self.fc(self.dropout(torch.cat(outputs, dim=1)))

class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Module initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)
