"""
An original implementation of sparsemax (Martins & Astudillo, 2016) is available at
https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/sparse_activations.py.
See `From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification, ICML 2016`
for detailed description.
"""
from dgl.base import ALL, is_all
from dgl.backend import astype
from dgl.heterograph_index import HeteroGraphIndex
import torch
from torch import Tensor
import dgl
from dgl.sparse import _gspmm, _gsddmm
# from dgl.ops import gspmm, gsddmm
from torch.autograd import Function


def _neighbor_sort(scores:Tensor, end_n_ids:Tensor, in_degrees:Tensor, cum_in_degrees:Tensor):
    num_nodes, max_in_degree = in_degrees.size(0), int(in_degrees.max().item())
    
    # compute the index for dense score matrix with size (N x D_{max})
    index = torch.arange(end_n_ids.size(0), dtype=torch.long, device=scores.device)
    index = (index - cum_in_degrees[end_n_ids]) + (end_n_ids * max_in_degree)
    index = index.long()

    dense_scores = scores.new_full((num_nodes * max_in_degree, ), torch.finfo(scores.dtype).min)
    dense_scores[index] = scores
    dense_scores = dense_scores.view(num_nodes, max_in_degree)

    sorted_dense_scores, _ = dense_scores.sort(dim=-1, descending=True)
    cumsum_sorted_dense_scores = sorted_dense_scores.cumsum(dim=-1).view(-1)
    sorted_dense_scores = sorted_dense_scores.view(-1)
    arange_vec = torch.arange(1, max_in_degree + 1, dtype=torch.long, device=end_n_ids.device)
    arange_vec = torch.repeat_interleave(arange_vec.view(1, -1), num_nodes, dim=0).view(-1)

    valid_mask = (sorted_dense_scores != torch.finfo(scores.dtype).min)
    sorted_scores = sorted_dense_scores[valid_mask]
    cumsum_sorted_scores = cumsum_sorted_dense_scores[valid_mask]
    arange_vec = arange_vec[valid_mask]

    return sorted_scores, cumsum_sorted_scores, arange_vec


def _threshold_and_support(gidx:HeteroGraphIndex, scores:Tensor, end_n_ids:Tensor):
    in_degrees = _gspmm(gidx, "copy_rhs", "sum", None, torch.ones_like(scores))[0]
    cum_in_degrees = torch.cat([in_degrees.new_zeros(1), in_degrees.cumsum(dim=0)[:-1]], dim=0)
    
    sorted_scores, cumsum_scores, rhos = _neighbor_sort(scores, end_n_ids, in_degrees, cum_in_degrees)
    cumsum_scores = cumsum_scores - 1.
    support = rhos * sorted_scores > cumsum_scores

    gidx.reverse()
    support_size = _gspmm(gidx, "copy_rhs", "sum", None, support.float())[0] #
    gidx.reverse()
    support_size = support_size.long()
    idx = support_size + cum_in_degrees - 1
    tau = cumsum_scores.gather(0, idx.long())
    tau /= support_size.to(scores.dtype)

    return tau, support_size


class EdgeSparsemax(Function):
    
    @staticmethod
    def forward(ctx, gidx:HeteroGraphIndex, scores:Tensor, 
                eids:Tensor, end_n_ids:Tensor, norm_by:str):
        if not is_all(eids):
            gidx = gidx.edge_subgraph([eids], True).graph
        if norm_by == "src":
            gidx = gidx.reverse()

        # use feat - max(feat) for numerical stability.
        scores = scores.float()
        scores_max = _gspmm(gidx, "copy_rhs", "max", None, scores)[0]
        scores = _gsddmm(gidx, "sub", scores, scores_max, "e", "v")

        tau, supp_size = _threshold_and_support(gidx, scores, end_n_ids)
        out = torch.clamp(_gsddmm(gidx, "sub", scores, tau, "e", "v"), min=0)
        ctx.backward_cache = gidx
        ctx.save_for_backward(supp_size, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        gidx = ctx.backward_cache
        supp_size, out = ctx.saved_tensors
        grad_in = grad_out.clone()
        grad_in[out == 0] = 0

        v_hat = _gspmm(gidx, "copy_rhs", "sum", None, grad_in)[0] / supp_size.to(out.dtype)
        grad_in_modify = _gsddmm(gidx, "sub", grad_in, v_hat, "e", "v")
        grad_in = torch.where(out != 0, grad_in_modify, grad_in)

        return None, grad_in, None, None, None


def edge_sparsemax(graph:dgl.DGLGraph, logits, eids=ALL, norm_by="dst"):
    row, _ = graph.all_edges(order="srcdst")
    if not is_all(eids):
        eids = astype(eids, graph.idtype)
        row = row[eids]
    return EdgeSparsemax.apply(graph._graph, logits,
                               eids, row, norm_by)


if __name__ == "__main__":
    from test_utils import fake_data
    g1 = fake_data(2, 2)
    g2 = fake_data(3, 2)
    g = dgl.batch([g1, g2])

    # scores = torch.randn((g.num_edges(),))
    scores = torch.arange(1, g.num_edges() + 1)
    res = edge_sparsemax(g, scores)
    print(res)
