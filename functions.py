"""
An original implementation of sparsemax (Martins & Astudillo, 2016) is available at
https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/sparse_activations.py.
See `From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification, ICML 2016`
for detailed description.

Here we implement a graph-edge version of sparsemax where we perform sparsemax for all edges
with the same node as end-node in graphs.
"""
import dgl
import torch
from dgl.backend import astype
from dgl.base import ALL, is_all
from dgl.heterograph_index import HeteroGraphIndex
from dgl.sparse import _gsddmm, _gspmm
from torch import Tensor
from torch.autograd import Function


def _neighbor_sort(scores:Tensor, end_n_ids:Tensor, in_degrees:Tensor, cum_in_degrees:Tensor):
    """Sort edge scores for each node"""
    num_nodes, max_in_degree = in_degrees.size(0), int(in_degrees.max().item())
    
    # Compute the index for dense score matrix with size (N x D_{max})
    # Note that the end_n_ids here is the end_node tensor in dgl graph,
    # which is not grouped by its node id (i.e. in this form: 0,0,1,1,1,...,N,N).
    # Thus here we first sort the end_node tensor to make it easier to compute
    # indexs in dense edge score matrix. Since we will need the original order
    # for following gspmm and gsddmm operations, we also keep the reverse mapping 
    # (the reverse_perm) here.
    end_n_ids, perm = torch.sort(end_n_ids)
    scores = scores[perm]
    _, reverse_perm = torch.sort(perm)

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

    return sorted_scores, cumsum_sorted_scores, arange_vec, reverse_perm


def _threshold_and_support_graph(gidx:HeteroGraphIndex, scores:Tensor, end_n_ids:Tensor):
    """Find the threshold for each node and its edges"""
    in_degrees = _gspmm(gidx, "copy_rhs", "sum", None, torch.ones_like(scores))[0]
    cum_in_degrees = torch.cat([in_degrees.new_zeros(1), in_degrees.cumsum(dim=0)[:-1]], dim=0)
    
    sorted_scores, cumsum_scores, rhos, reverse_perm = _neighbor_sort(scores, end_n_ids, 
                                                                      in_degrees, cum_in_degrees)
    cumsum_scores = cumsum_scores - 1.
    support = rhos * sorted_scores > cumsum_scores
    support = support[reverse_perm]

    support_size = _gspmm(gidx, "copy_rhs", "sum", None, support.float())[0]
    support_size = support_size.long().clamp(min=1)
    idx = support_size + cum_in_degrees - 1
    # mask invalid index, for example, if batch is not start from 0 or not continuous, it may result in negative index
    mask = idx < 0
    idx[mask] = 0
    tau = cumsum_scores.gather(0, idx.long())
    tau /= support_size.to(scores.dtype)

    return tau, support_size


class EdgeSparsemaxFunction(Function):
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

        # find threshold for each node and perform ReLU(u-t(u)) operation.
        tau, supp_size = _threshold_and_support_graph(gidx, scores, end_n_ids)
        out = torch.clamp(_gsddmm(gidx, "sub", scores, tau, "e", "v"), min=0)
        ctx.backward_cache = gidx
        ctx.save_for_backward(supp_size, out)
        torch.cuda.empty_cache()
        return out

    @staticmethod
    def backward(ctx, grad_out):
        gidx = ctx.backward_cache
        supp_size, out = ctx.saved_tensors
        grad_in = grad_out.clone()

        # grad for ReLU
        grad_in[out == 0] = 0

        # dL/dv_i = dL/do_i - 1/k \sum_{j=1}^k dL/do_j
        v_hat = _gspmm(gidx, "copy_rhs", "sum", None, grad_in)[0] / supp_size.to(out.dtype)
        grad_in_modify = _gsddmm(gidx, "sub", grad_in, v_hat, "e", "v")
        grad_in = torch.where(out != 0, grad_in_modify, grad_in)
        del gidx
        torch.cuda.empty_cache()
        
        return None, grad_in, None, None, None


def edge_sparsemax(graph:dgl.DGLGraph, logits, eids=ALL, norm_by="dst"):
    row, col = graph.all_edges(order="srcdst")
    assert norm_by in ["dst", "src"]
    end_n_ids = col if norm_by == "dst" else row
    if not is_all(eids):
        eids = astype(eids, graph.idtype)
        end_n_ids = end_n_ids[eids]
    return EdgeSparsemaxFunction.apply(graph._graph, logits,
                                       eids, end_n_ids, norm_by)


class EdgeSparsemax(torch.nn.Module):
    r"""
    Description
    -----------
    Compute edge sparsemax. For a node :math:`i`, edge sparsemax is an operation that computes 

    .. math::
      a_{ij} = \text{ReLU}(z_{ij} - \tau(\z_{i,:}))
    
    where :math:`z_{ij}` is a signal of edge :math:`j\rightarrow i`, also
    called logits in the context of sparsemax. :math:`\tau` is a function
    that can be found at the `From Softmax to Sparsemax <https://arxiv.org/pdf/1602.02068.pdf>`
    paper.

    Parameters
    ----------
    graph : DGLGraph
        The graph to perform edge sparsemax on.
    logits : torch.Tensor
        The input edge feature.
    eids : torch.Tensor or ALL, optional
        A tensor of edge index on which to apply edge sparsemax. If ALL, apply edge
        sparsemax on all edges in the graph. Default: ALL.
    norm_by : str, could be 'src' or 'dst'
        Normalized by source nodes of destination nodes. Default: `dst`.

    Returns
    -------
    Tensor
        Sparsemax value.
    """
    def __init__(self):
        super(EdgeSparsemax, self).__init__()
    
    def forward(self, graph, logits, eids=ALL, norm_by="dst"):
        return edge_sparsemax(graph, logits, eids, norm_by)


def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _threshold_and_support(input, dim=0):
    """Sparsemax building block: compute the threshold
    
    Parameters
    ----------
    input : Tensor
        any dimension
    dim : int
        dimension along which to apply the sparsemax
    
    Returns
    -------
        the threshold value
    """

    input_srt, _ = torch.sort(input, descending=True, dim=dim)
    input_cumsum = input_srt.cumsum(dim) - 1
    rhos = _make_ix_like(input, dim)
    support = rhos * input_srt > input_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = input_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(input.dtype)
    return tau, support_size


class SparsemaxFunction(Function):

    @staticmethod
    def forward(ctx, input, dim=0):
        """sparsemax: normalizing sparse transform (a la softmax)
        Parameters:
            input (Tensor): any shape
            dim: dimension along which to apply sparsemax
        Returns:
            output (Tensor): same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = _threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None


def sparsemax(input, dim=0):
    return SparsemaxFunction.apply(input, dim)


class Sparsemax(torch.nn.Module):
    def __init__(self, dim=0):
        super(Sparsemax, self).__init__()
        self.dim = dim

    def forward(self, input):
        return sparsemax(input, self.dim)
