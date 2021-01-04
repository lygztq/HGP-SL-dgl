# DGL Implementation of the HGP-SL Paper

This DGL example implements the GNN model proposed in the paper [Hierarchical Graph Pooling with Structure Learning](https://arxiv.org/pdf/1911.05954.pdf). 
The author's codes of implementation is in [here](https://github.com/cszhangzhen/HGP-SL)


Example implementor
----------------------
This example was implemented by [Tianqi Zhang](https://github.com/lygztq) during his Applied Scientist Intern work at the AWS Shanghai AI Lab.


The graph dataset used in this example 
---------------------------------------
The DGL's built-in LegacyTUDataset. This is a serial of graph kernel datasets for graph classification. We use 'DD', 'PROTEINS', 'NCI1', 'NCI109', 'Mutagenicity' and 'ENZYMES' in this HGP-SL implementation. All these datasets are randomly splited to train, validation and test set with ratio 0.8, 0.1 and 0.1.

NOTE: Since there is no data attributes in some of these datasets, we use node_id (in one-hot vector whose length is the max number of nodes across all graphs) as the node feature. Also note that the node_id in some datasets is not unique (e.g. a graph may has two nodes with the same id).

DD
- NumGraphs: 1178
- AvgNodesPerGraph: 284.32
- AvgEdgesPerGraph: 715.66
- NumFeats: 89
- NumClasses: 2

PROTEINS
- NumGraphs: 1113
- AvgNodesPerGraph: 39.06
- AvgEdgesPerGraph: 72.82
- NumFeats: 1
- NumClasses: 2

NCI1
- NumGraphs: 4110
- AvgNodesPerGraph: 29.87
- AvgEdgesPerGraph: 32.30
- NumFeats: 37
- NumClasses: 2

NCI109
- NumGraphs: 4127
- AvgNodesPerGraph: 29.68
- AvgEdgesPerGraph: 32.13
- NumFeats: 38
- NumClasses: 2

Mutagenicity
- NumGraphs: 4337
- AvgNodesPerGraph: 30.32
- AvgEdgesPerGraph: 30.77
- NumFeats: 14
- NumClasses: 2

ENZYMES
- NumGraphs: 600
- AvgNodesPerGraph: 32.63
- AvgEdgesPerGraph: 62.14
- NumFeats: 18
- NumClasses: 6

How to run example files
--------------------------------
In the HGP-SL-DGL folder, run

```bash
python main.py --dataset ${your_dataset_name_here}
```

If want to use a GPU, run

```bash
python main.py --device ${your_device_id_here} --dataset ${your_dataset_name_here}
```

Performance
-------------------------

|                   | Mutagenicity | NCI109      | NCI1        |
| ----------------- | ------------ | ----------- | ----------- |
| Reported in Paper | 82.15(0.58)  | 80.67(1.16) | 78.45(0.77) |
| Author's Code     | 79.68(1.68)  | 73.86(1.72) | 76.29(2.14) |
| Ours              | 78.67(1.63)  | 72.80(2.96) | 74.06(2.34) |
