## DGL Implementation of the MixHopGCN Paper

This DGL example implements the GNN model proposed in the paper [MixHop: Higher-Order Graph Convolution Architectures via Sparsified Neighborhood Mixing](https://arxiv.org/abs/1905.00067). The author's codes of implementation is in [here](https://github.com/samihaija/mixhop).

## Example implementor

This example was implemented by [xnuohz](https://github.com/xnuohz) during his intern at the AWS Shanghai AI Lab.

### Requirements
The codebase is implemented in Python 3.6.11. package versions used for devalopment are just below.

```
dgl 0.5.2
numpy 1.19.4
pandas 1.1.4
tqdm 4.53.0
torch 1.7.0
```

### The graph dataset used in this example

The DGL's built-in Cora, Pubmed and Citeseer GraphDataset. Dataset summary:

| Datset | Nodes | Edges | Feats | Classes | Train | Val | Test |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Citeseer | 3,327 | 9,228 | 3,703 | 6 | 120 | 500 | 1000 |
| Cora | 2,708 | 10,556 | 1,433 | 7 | 140 | 500 | 1000 |
| Pubmed | 19,717 | 88,651 | 500 | 3 | 60 | 500 | 1000 |

### Usage

###### Dataset options
```
--dataset     string     The graph dataset name.     Default is 'Cora'.
```

###### Model options
```
--epochs           INT     Number of training epochs.      Default is 2000.
--early-stopping   INT     Early stopping rounds.          Default is 200.
--lr               FLOAT   Adam optimizer learning rate.   Default is 0.05.
--lamb             FLOAT   L2 regularization coefficient.  Default is 0.05.
--gclayers         LST     MixHop GC Layer sizes.          Default is [60, 60].
--p                LST     Powers of adjacency matrix.     Default is [0, 1, 2].
```

###### Examples

The following commands learn a neural nwtwork and predict on the test set.
Training a MixHop model on the default dataset.
```bash
$ python src/main.py
```
Training a model for a 200 epochs and a 10 early stopping.
```bash
$ python src/main.py --epochs 200 --early-stopping 10
```
Training a model with different learning rate and regularization
```bash
$ python src/main.py --lr 0.001 --lamb 0.1
```
Training a model with different model params
```bash
$ python src/main.py --gclayers 200 200 200 --p 2 4 6
```

### Performance

| Datset | Cora | Pubmed | Citeseer |
| :-: | :-: | :-: | :-: |
| Accuracy(original paper) | 0.818 | 0.800 | 0.714 |
| Accuracy(pyg) | 0.588 | 0.655 | 0.301 |
| Accuracy(DGL) | 0.788 | 0.776 | 0.645 |

### Ref

[Example Guide](https://github.com/zhjwy9343/MVP4ModelExample)