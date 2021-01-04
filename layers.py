import logging

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import AvgPooling, GraphConv, MaxPooling
from dgl.ops import edge_softmax
from torch import Tensor
from torch.nn import Parameter

from functions import edge_sparsemax
from utils import get_batch_id, topk


def _khop_union_graph(graph, k):
    # pyg use spspmm here, however dgl has no
    # efficient way to implement this function.
    # Stub here.
    pass


class WeightedGraphConv(GraphConv):
    """GraphConv with edge weights on homogeneous graphs"""
    def forward(self, graph:DGLGraph, n_feat, e_feat=None):
        if e_feat is None:
            return super(WeightedGraphConv, self).forward(graph, n_feat)
        
        with graph.local_scope():
            src_norm = torch.pow(graph.out_degrees().float().clamp(min=1), -0.5)
            src_norm = src_norm.view(-1, 1)
            dst_norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
            dst_norm = dst_norm.view(-1, 1)
            n_feat = n_feat * src_norm
            graph.ndata["h"] = n_feat
            graph.edata["e"] = e_feat
            graph.update_all(fn.src_mul_edge("h", "e", "m"),
                             fn.sum("m", "h"))
            n_feat = graph.ndata.pop("h")
            n_feat = n_feat * dst_norm
            return n_feat


class NodeInfoScoreLayer(nn.Module):
    r"""
    Description
    -----------
    Compute a score for each node for sort-pooling. The score of each node
    is computed via the absolute difference of its first-order random walk
    result and its features.

    Parameters
    ----------
    graph : DGLGraph
        The graph to perform this operation.
    feat : torch.Tensor
        The node features
    e_feat : torch.Tensor, optional
        The edge features. Default: :obj:`None`
    
    Returns
    -------
    Tensor
        Score for each node.
    """
    def __init__(self, sym_norm:bool=True):
        super(NodeInfoScoreLayer, self).__init__()
        self.sym_norm = sym_norm

    def forward(self, graph:dgl.DGLGraph, feat:Tensor, e_feat:Tensor):
        with graph.local_scope():
            if self.sym_norm:
                src_norm = torch.pow(graph.out_degrees().float().clamp(min=1), -0.5)
                src_norm = src_norm.view(-1, 1).to(feat.device)
                dst_norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
                dst_norm = dst_norm.view(-1, 1).to(feat.device)

                src_feat = feat * src_norm
                
                graph.ndata["h"] = src_feat
                graph.edata["e"] = e_feat
                graph.update_all(fn.src_mul_edge("h", "e", "m"), fn.sum("m", "h"))
                
                dst_feat = graph.ndata.pop("h") * dst_norm
                feat = feat - dst_feat
            else:
                dst_norm = 1. / graph.in_degrees().float().clamp(min=1)
                dst_norm = dst_norm.view(-1, 1)

                graph.ndata["h"] = feat
                graph.edata["e"] = e_feat
                graph.update_all(fn.src_mul_edge("h", "e", "m"), fn.sum("m", "h"))

                feat = feat - dst_norm * graph.ndata.pop("h")

            score = torch.sum(torch.abs(feat), dim=1)
            return score


class HGPSLPool(nn.Module):
    r"""

    Description
    -----------
    The HGP-SL pooling layer from 
    `Hierarchical Graph Pooling with Structure Learning <https://arxiv.org/pdf/1911.05954.pdf>`

    Parameters
    ----------
    in_feat : int
        The number of input node feature's channels
    ratio : float, optional
        Pooling ratio. Default: 0.8
    sample : bool, optional
        Whether use k-hop union graph to increase efficiency. 
        Currently we only support full graph. Default: :obj:`False`
    sym_score_norm : bool, optional
        Use symmetric norm for adjacency or not. Default: :obj:`True`
    sparse : bool, optional
        Use edge sparsemax instead of edge softmax. Default: :obj:`True`
    sl : bool, optional
        Use structure learining module or not. Default: :obj:`True`
    lamb : float, optional
        The lambda parameter as weight of raw adjacency as described in the
        HGP-SL paper. Default: 1.0
    negative_slop : float, optional
        Negative slop for leaky_relu. Default: 0.2
    
    Returns
    -------
    DGLGraph
        The pooled graph.
    torch.Tensor
        Node features
    torch.Tensor
        Edge features
    torch.Tensor
        Permutation index
    """
    def __init__(self, in_feat:int, ratio=0.8, sample=False, 
                 sym_score_norm=True, sparse=True, sl=True,
                 lamb=1.0, negative_slop=0.2, k_hop=3):
        super(HGPSLPool, self).__init__()
        self.in_feat = in_feat
        self.ratio = ratio
        self.sample = sample
        self.sparse = sparse
        self.sl = sl
        self.lamb = lamb
        self.negative_slop = negative_slop
        self.k_hop = k_hop

        # TODO: support sample method
        if self.sample:
            logging.warning("Currently we do not support sample method,"
                            "use complete graph instead...")
            self.sample = False

        self.att = Parameter(torch.Tensor(1, self.in_feat * 2))
        self.calc_info_score = NodeInfoScoreLayer(sym_norm=sym_score_norm)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.att.data)

    def forward(self, graph:DGLGraph, feat:Tensor, e_feat=None):
        # top-k pool first
        if e_feat is None:
            e_feat = torch.ones((graph.number_of_edges(),), 
                                dtype=feat.dtype, device=feat.device)
        x_score = self.calc_info_score(graph, feat, e_feat)
        batch_num_nodes = graph.batch_num_nodes()
        perm, next_batch_num_nodes = topk(x_score, self.ratio, 
                                          get_batch_id(batch_num_nodes),
                                          batch_num_nodes)
        feat = feat[perm]
        graph.edata["e"] = e_feat
        pool_graph = dgl.node_subgraph(graph, perm)
        e_feat = pool_graph.edata.pop("e")
        pool_graph.set_batch_num_nodes(next_batch_num_nodes)

        # no structure learning layer, directly return.
        if not self.sl:
            return pool_graph, feat, e_feat, perm

        # Structure Learning
        if self.sample:
            # A fast mode for large graphs.
            # In large graphs, learning the possible edge weights between each
            # pair of nodes is time consuming. To accelerate this process,
            # we sample it's K-Hop neighbors for each node and then learn the
            # edge weights between them.

            # Note: k-hop graph here is from original graph, not pooled graph.
            pass

        else: 
            # Learning the possible edge weights between each pair of
            # nodes in the pooled subgraph, relative slower.

            # first construct complete graphs for all graph in the batch
            # use dense to build, then transform to sparse.
            # maybe there's more efficient way?
            batch_num_nodes = next_batch_num_nodes
            block_begin_idx = torch.cat([batch_num_nodes.new_zeros(1), 
                batch_num_nodes.cumsum(dim=0)[:-1]], dim=0)
            block_end_idx = batch_num_nodes.cumsum(dim=0)
            dense_adj = torch.zeros((pool_graph.num_nodes(),
                                    pool_graph.num_nodes()),
                                    dtype=torch.float, 
                                    device=feat.device)
            for idx_b, idx_e in zip(block_begin_idx, block_end_idx):
                dense_adj[idx_b:idx_e, idx_b:idx_e] = 1.
            row, col = torch.nonzero(dense_adj).t().contiguous()

            # compute weights for node-pairs
            weights = (torch.cat([feat[row], feat[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop)
            dense_adj[row, col] = weights

            # add pooled graph structure to weight matrix
            pool_row, pool_col = pool_graph.all_edges()
            dense_adj[pool_row, pool_col] += self.lamb * e_feat
            weights = dense_adj[row, col]

            # edge softmax/sparsemax
            complete_graph = dgl.graph((row, col))
            if self.sparse:
                weights = edge_sparsemax(complete_graph, weights)
            else:
                weights = edge_softmax(complete_graph, weights)

            # get new e_feat and graph structure, clean up.
            dense_adj[row, col] = weights
            weights = weights[weights != 0]
            e_feat = weights
            row, col = torch.nonzero(dense_adj).t().contiguous()
            pool_graph = dgl.graph((row, col))
            pool_graph.set_batch_num_nodes(next_batch_num_nodes)
            del dense_adj
            torch.cuda.empty_cache()

        return pool_graph, feat, e_feat, perm


class ConvPoolReadout(torch.nn.Module):
    """A helper class. (GraphConv -> Pooling -> Readout)"""
    def __init__(self, in_feat:int, out_feat:int, pool_ratio=0.8,
                 sample:bool=False, sparse:bool=True, sl:bool=True,
                 lamb:float=1., pool:bool=True):
        super(ConvPoolReadout, self).__init__()
        self.use_pool = pool
        self.conv = WeightedGraphConv(in_feat, out_feat)
        if pool:
            self.pool = HGPSLPool(out_feat, ratio=pool_ratio, sparse=sparse,
                                  sample=sample, sl=sl, lamb=lamb)
        else:
            self.pool = None
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

    def forward(self, graph, feature, e_feat=None):
        out = F.relu(self.conv(graph, feature, e_feat))
        if self.use_pool:
            graph, out, e_feat, _ = self.pool(graph, out, e_feat)
        readout = torch.cat([self.avgpool(graph, out), self.maxpool(graph, out)], dim=-1)
        return graph, out, e_feat, readout
