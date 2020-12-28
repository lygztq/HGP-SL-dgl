import torch
import dgl


def generate_adj_matrix(num_nodes, symmetric=True, sparsity=0.5, connected=True):
    """Generate a adjacency matrix (in dense format) randomly.
    
    Args:
        num_nodes (int): The number of nodes in this graph.
        symmetric (bool, optional): If set :obj:`True`, will generate a 
            symmetric adjacency matrix. (default: :obj:`True`)
        sparsity (float, optional): The parameter that controls the sparsity
            of the generated graph. This value should in (0, 1]. (default: 0.5)
        connected (bool, optional): If set :obj:`True`, will generate a 
            connected graph. (default: :obj:`True`)
    """
    A = torch.rand((num_nodes, num_nodes), dtype=torch.float)
    if symmetric:
        A = 0.5 * (A + A.t())
    A = (A < (1 - sparsity)).long() * (1 - torch.eye(num_nodes))
    if connected:
        start = torch.randperm(num_nodes)
        end = torch.cat([start[1:], start.new_full(size=[1], fill_value=start[0].item())])
        A[start, end] = 1
        A[end, start] = 1
    return A


def fake_data(num_nodes, num_features, sparsity=0.5, one_hot=False, 
              directed=False, weighted=False, with_label=False, label_dim=None,
              add_self_loop=False, device="cpu"):
    r"""Generate a fake graph data for test or debug. For DGL, the node feature
    is named as ndata['x'], the node label is named as ndata['y'] and edge weight is named as 
    ndata['weight']
    Args:
        num_nodes (int): The number of nodes in this graph.
        num_features (int): The dimension of node features.
        sparsity (float, optional): The parameter that controls the sparsity
            of the generated graph. This value should in (0, 1]. (default: 0.5)
        one_hot (bool, optional): If set :obj:`True`, generate one-hot node feature.
            (default: :obj:`False`)
        directed (bool, optional): If set :obj:`True`, generate asymmetric adjacency
            matrix. (default: :obj:`False`)
        weighted (bool, optional): If set :obj:`True`, generate edge with weight.
            (default: :obj:`False`)
        with_label (bool, optional): If set :obj:`True`, generate label. (default: :obj:`False`)
        label_dim (int, optional): If :obj:`with_label` is :obj:`True`, this argument must be given.
            The dimension of label. (default: :obj:`None`)
        add_self_loop (bool, optional): If set :obj:`True`, add self-loop to all nodes.
            (default: :obj:`False`)
        device (str, optional): The device where this graph data in. (default: :obj:`"cpu"`)
    """
    device = torch.device(device)
    x = torch.randn((num_nodes, num_features), device=device)
    if one_hot:
        x = (torch.max(x, axis=-1, keepdim=True)[0] == x).float()
    A = generate_adj_matrix(num_nodes, not directed, sparsity=sparsity)
    if add_self_loop:
        A += torch.eye(num_nodes)
    edge_idx = torch.nonzero(A == 1).t().contiguous()
    edge_idx = edge_idx.to(device)

    y, edge_attr = None, None
    if weighted:
        edge_attr = torch.abs(torch.randn(edge_idx.shape[-1], device=device))
    if with_label:
        if label_dim is None:
            raise ValueError("The argument 'label_dim' must be given when generate data with label.")
        y = torch.randint(label_dim, size=[num_nodes], dtype=torch.int, device=device)
    
    g = dgl.graph(data=(edge_idx[0], edge_idx[1]), num_nodes=num_nodes)
    g.ndata["x"] = x
    if with_label:
        g.ndata["y"] = y
    if weighted:
        g.edata["weight"] = edge_attr
    
    return g
