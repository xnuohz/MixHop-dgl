""" The main file to train a Simplified MixHopGCN model using a full graph """

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import dgl.function as fn

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from tqdm import trange

class MixHopConv(nn.Module):
    """
    One layer of MixHopGCN.
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 p=[0, 1, 2],
                 dropout=0,
                 activation=None,
                 batchnorm=False):
        super(MixHopConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p = p
        self.activation = activation
        self.batchnorm = batchnorm
        self.fc = nn.Linear(out_dim * len(p), out_dim)

        # define dropout layer
        self.dropout = nn.Dropout(dropout)

        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        
        # define weight dict for each power j
        self.weights = nn.ModuleDict({
            str(j): nn.Linear(in_dim, out_dim) for j in p
        })

    def forward(self, graph, feats):
        with graph.local_scope():
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1)
            max_j = max(self.p) + 1
            outputs = []
            for j in range(max_j):
                feats = feats * norm
                graph.ndata['h'] = feats
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                feats = graph.ndata.pop('h')
                feats = feats * norm

                if j in self.p:
                    output = self.weights[str(j)](feats)
                    if self.activation is not None:
                        output = self.activation(output)
                    outputs.append(output)
            
            final = self.fc(torch.cat(outputs, dim=1))

            final = self.bn(final)
            final = self.dropout(final)

            return final

class MixHopGCN(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim, 
                 out_dim,
                 num_layers=2,
                 p=[0, 1, 2],
                 dropout=0.0,
                 activation=None,
                 batchnorm=False):
        super(MixHopGCN, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.p = p
        self.dropout = dropout
        self.activation = activation
        self.batchnorm = batchnorm

        self.layers = nn.ModuleList()

        # Input layer and initial q for output layer
        self.n_segs = int(self.hid_dim / self.out_dim)

        if self.hid_dim % self.out_dim != 0:
            print('Wasted columns: {} out of {}'.format(self.hid_dim % self.out_dim, self.hid_dim))

        self.q = nn.Parameter(torch.tensor(np.random.rand(self.n_segs), requires_grad=True))
        # self.q = nn.Softmax(dim=0)(self.q)

        self.layers.append(MixHopConv(self.in_dim,
                                      self.hid_dim,
                                      p=self.p,
                                      dropout=self.dropout,
                                      activation=self.activation,
                                      batchnorm=self.batchnorm))
        
        # Hidden layers with n - 1 MixHopConv layers
        for i in range(self.num_layers - 2):
            self.layers.append(MixHopConv(self.hid_dim,
                                          self.hid_dim,
                                          p=self.p,
                                          dropout=self.dropout,
                                          activation=self.activation,
                                          batchnorm=self.batchnorm))

    def forward(self, graph, feats):
        for layer in self.layers:
            feats = layer(graph, feats)

        output = 0
        for k in range(self.n_segs):
            segment = feats[:, k * self.out_dim : (k + 1) * self.out_dim]
            output = segment * self.q[k] + output
        
        return output

def main(args):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # Load from DGL dataset
    if args.dataset == 'Cora':
        dataset = CoraGraphDataset(raw_dir=args.raw_dir)
    elif args.dataset == 'Citeseer':
        dataset = CiteseerGraphDataset(raw_dir=args.raw_dir)
    elif args.dataset == 'Pubmed':
        dataset = PubmedGraphDataset(raw_dir=args.raw_dir)
    else:
        raise ValueError('Dataset {} is invalid.'.format(args.raw_dir))

    graph = dataset[0]
    graph = dgl.add_self_loop(graph)

    # check cuda
    if args.gpu > 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    # retrieve the number of classes
    n_classes = dataset.num_classes

    # retrieve labels of ground truth
    labels = graph.ndata.pop('label').to(device).long()

    # Extract node features
    feats = graph.ndata.pop('feat').to(device)
    n_features = feats.shape[-1]

    # retrieve masks for train/validation/test
    train_mask = graph.ndata.pop('train_mask')
    val_mask = graph.ndata.pop('val_mask')
    test_mask = graph.ndata.pop('test_mask')

    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze().to(device)
    val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze().to(device)
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze().to(device)

    graph.to(device)

    # Step 2: Create model =================================================================== #
    mixhopgcn_model = MixHopGCN(in_dim=n_features,
                                hid_dim=args.hid_dim,
                                out_dim=n_classes,
                                num_layers=args.num_layers,
                                p=args.p,
                                dropout=args.dropout,
                                activation=torch.tanh,
                                batchnorm=True)
    
    mixhopgcn_model = mixhopgcn_model.to(device)

    # Step 3: Create training components ===================================================== #
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(mixhopgcn_model.parameters(), lr=args.lr, weight_decay=args.lamb)

    # Step 4: training epoches =============================================================== #
    acc = 0
    no_improvement = 0
    epochs = trange(args.epochs, desc='Accuracy & Loss')
    
    for _ in epochs:

        # Training and validation using a full graph
        mixhopgcn_model.train()

        logits = mixhopgcn_model.forward(graph, feats)

        # compute loss
        train_loss = loss_fn(logits[train_idx], labels[train_idx])
        train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)

        valid_loss = loss_fn(logits[val_idx], labels[val_idx])
        valid_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)

        # backward
        opt.zero_grad()
        train_loss.backward()
        opt.step()

        # Print out performance
        epochs.set_description('Train Acc {:.4f} | Train Loss {:.4f} | Val Acc {:.4f} | Val loss {:.4f}'.format(
            train_acc, train_loss.item(), valid_acc, valid_loss.item()))
        
        if valid_acc < acc:
            no_improvement += 1
            if no_improvement == args.early_stopping:
                print('Early stop.')
                break
        else:
            no_improvement = 0
            acc = valid_acc

    # Test with mini batch after all epoch
    mixhopgcn_model.eval()

    # forward
    logits = mixhopgcn_model.forward(graph, feats)

    # compute loss
    test_loss = loss_fn(logits[test_idx], labels[test_idx])
    test_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)

    print("Test Acc {:.4f} | Test loss {:.4f}".format(test_acc, test_loss.item()))

if __name__ == "__main__":
    """
    MixHop Model Parameters
    """
    parser = argparse.ArgumentParser(description='MixHop GCN')

    # data source params
    parser.add_argument('--dataset', type=str, default='Cora', help='Name of dataset.')
    parser.add_argument('--raw_dir', type=str, default='./data', help='Path of dataset.')
    # cuda params
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index. Default: -1, using CPU.')
    # model params
    parser.add_argument('--epochs', type=int, default=2000, help='Traning epochs.')
    parser.add_argument('--early-stopping', type=int, default=200, help='Early stopping.')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate.')
    parser.add_argument('--lamb', type=float, default=5e-2, help='L2 reg.')
    parser.add_argument("--hid_dim", type=int, default=200, help='Hidden layer dimensionalities.')
    parser.add_argument("--num_layers", type=int, default=4, help='Number of GNN layers.')
    parser.add_argument("--dropout", type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--p', nargs='+', type=int, help='Powers list of adjacency matrix.')

    parser.set_defaults(p=[0, 1, 2])

    args = parser.parse_args()
    print(args)

    main(args)
