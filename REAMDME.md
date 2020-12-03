## MixHop
A DGL implementation of "MixHop: Higher-Order Graph Convolution Architectures via Sparsified Neighborhood Mixing" (ICML 2019).

### Requirements
The codebase is implemented in Python 3.6.11. package versions used for devalopment are just below.

```
dgl 0.5.2
numpy 1.19.4
pandas 1.1.4
tqdm 4.53.0
torch 1.7.0
texttable 1.6.3
```

### MixHopConv

```
class MixHopConv(in_feats, out_feats, p=[0, 1, 2], bias=True, activation=None)
```

MixHop Graph Convolutional layer from paper [MixHop: Higher-Order Graph Convolution Architectures via Sparsified Neighborhood Mixing](https://arxiv.org/abs/1905.00067)

$$ H^{(i + 1)} = ||_{j \in P} \sigma(\hat{A}^j H^{(i)} W_j^{(i)}) $$

Where $\hat{A} = D^{-\frac{1}{2}}(A + I_n)D^{-\frac{1}{2}}$. The graph input is expected to have self-loop edges added.

#### Parameters:
* in_feats(int) - Number of input features; i.e, the number of dimensions of $H^{(i)}$.
* out_feats(int) - Number of output features; i.e, the number of dimensions of $H^{(i + 1)}.$
* p(list) - List of integer adjacency powers; i.e, the times of the adjacency matrix multiplied.
* bias(bool) - If True, adds a learnable bias to the output. Default: True.
* activation(callable activation function/layer or None, optional) - If not None, applies an activation function to the updated node features.

```
forward(graph, feat)
```
Compute MixHop GC layer

#### Parameters:
* graph(DGLGraph) - The graph; i.e, for Node Classification.
* feat(torch.Tensor) - The input feature of the graph with shape $(N, D_{in})$ where $D_{in}$ is size of input feature, $N$ is the number of nodes in the graph.

#### Returns:
The output feature of shape $(N, D_{out})$ where $D_{out}$ is size of output feature.

#### Return type:
torch.Tensor

### Usage

```
python src/main.py --dataset [dataset] --p 0 1 2 --epochs 200
```

### Dataset

| Datset | nodes | edges | features | c | train | val | test |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Citeseer | 3,327 | 9,228 | 3,703 | 6 | 120 | 500 | 1000 |
| Cora | 2,708 | 10,556 | 1,433 | 7 | 140 | 500 | 1000 |
| Pubmed | 19,717 | 88,651 | 500 | 3 | 60 | 500 | 1000 |

### Results(deafult)

* (v1) (gc+fc+dropout)s + (fc)s + softmax
* (v2) (gc+tanh+bn)s + split softmax

| Datset | Cora | Pubmed | Citeseer |
| :-: | :-: | :-: | :-: |
| Accuracy(original paper) | 0.818 | 0.800 | 0.714 |
| Accuracy(pyg) | 0.588 | 0. | 0. |
| Accuracy(DGL) | 0.788 | 0.762 | 0.602 |

### Ref

* [tensorflow](https://github.com/samihaija/mixhop)
* [pytorch](https://github.com/benedekrozemberczki/MixHop-and-N-GCN)
