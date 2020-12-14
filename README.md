# DGL Implementation of MixHop

This DGL example implements the GNN model proposed in the paper [MixHop: Higher-Order Graph Convolution Architectures via Sparsified Neighborhood Mixing](https://arxiv.org/abs/1905.00067). For the original implementation, see [here](https://github.com/samihaija/mixhop).

## Example implementor

This example was implemented by [xnuohz](https://github.com/xnuohz) during his intern at the AWS Shanghai AI Lab.

### Requirements
The codebase is implemented in Python 3.6. For version requirement of packages, see below.

```
dgl 0.5.2
numpy 1.19.4
pandas 1.1.4
tqdm 4.53.0
torch 1.7.0
```

### The graph datasets used in this example

The DGL's built-in Cora, Pubmed and Citeseer GraphDataset. Dataset summary:

| Dataset | Nodes | Edges | Feats | Classes | # Train Nodes | # Val Nodes | # Test Nodes |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Citeseer | 3,327 | 9,228 | 3,703 | 6 | 120 | 500 | 1000 |
| Cora | 2,708 | 10,556 | 1,433 | 7 | 140 | 500 | 1000 |
| Pubmed | 19,717 | 88,651 | 500 | 3 | 60 | 500 | 1000 |

### Usage

###### Dataset options
```
--dataset          str     The graph dataset name.             Default is 'Cora'.
```

###### GPU options
```
--gpu              int     GPU index.                          Default is -1, using CPU.
```

###### Model options
```
--epochs           int     Number of training epochs.          Default is 2000.
--early-stopping   int     Early stopping rounds.              Default is 200.
--lr               float   Adam optimizer learning rate.       Default is 0.5.
--lamb             float   L2 regularization coefficient.      Default is 0.0005.
--step-size        int     Period of learning rate decay.      Default is 40.
--gamma            float   Factor of learning rate decay.      Default is 0.01.
--hid-dim          int     Hidden layer dimensionalities.      Default is 60.
--num-layers       int     Number of GNN layers.               Default is 4.
--input-dropout    float   Dropout applied at input layer.     Default is 0.7.
--layer-dropout    float   Dropout applied at hidden layers.   Default is 0.9.
--p                list    Powers list of adjacency matrix.    Default is [0, 1, 2].
```

###### Examples

The following commands learn a neural network and predict on the test set.
Training a MixHop model on the default dataset.
```bash
$ python main.py
```
Train a model for 200 epochs and perform an early stop if the validation accuracy stops getting improved for 10 epochs.
```bash
$ python main.py --epochs 200 --early-stopping 10
```
Train a model with a different learning rate and regularization coefficient.
```bash
$ python main.py --lr 0.001 --lamb 0.1
```
Train a model with different model hyperparameters.
```bash
$ python main.py --num-layers 6 --p 2 4 6
```
Or just run the shell scripts which follow the original hyperparameters.
```bash
# Cora:
$ bash train_cora.sh

# Citeseer:
$ bash train_citeseer.sh

# Pubmed:
$ bash train_pubmed.sh
```

### Performance

| Dataset | Cora | Pubmed | Citeseer |
| :-: | :-: | :-: | :-: |
| Accuracy(original paper) | 0.818 | 0.800 | 0.714 |
| Accuracy(TF) | 0.816 | 0.789 | 0.712 |
| Accuracy(DGL) | 0.809 | 0.785 | 0.705 |