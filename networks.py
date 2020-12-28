import dgl
import torch
from dgl.nn import GraphConv, AvgPooling, MaxPooling
import torch.nn.functional as F

from layers import HGPSLPool, WeightedGraphConv


class Model(torch.nn.Module):
    def __init__(self, in_feat:int, out_feat:int, hid_feat:int,
                 dropout:float=0.0, pooling_ratio:float=0.5,
                 sample:bool=False, sparse:bool=True, sl:bool=True,
                 lamb:float=1.):
        super(Model, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.hid_feat = hid_feat
        self.dropout = dropout
        self.pooling_ratio = pooling_ratio
        
        self.conv1 = WeightedGraphConv(in_feat, hid_feat)
        self.conv2 = WeightedGraphConv(hid_feat, hid_feat)
        self.conv3 = WeightedGraphConv(hid_feat, hid_feat)

        self.pool1 = HGPSLPool(hid_feat, ratio=pooling_ratio, 
                               sample=sample, sparse=sparse,
                               sl=sl, lamb=lamb)
        self.pool2 = HGPSLPool(hid_feat, ratio=pooling_ratio, 
                               sample=sample, sparse=sparse,
                               sl=sl, lamb=lamb)

        self.lin1 = torch.nn.Linear(hid_feat * 2, hid_feat)
        self.lin2 = torch.nn.Linear(hid_feat, hid_feat // 2)
        self.lin3 = torch.nn.Linear(hid_feat // 2, self.out_feat)
 
        self.avg_pool = AvgPooling()
        self.max_pool = MaxPooling()

    def forward(self, graph, n_feat):
        final_readout = None

        n_feat = F.relu(self.conv1(graph, n_feat))
        graph, n_feat, e_feat, _ = self.pool1(graph, n_feat, None)
        final_readout = torch.cat([self.avg_pool(graph, n_feat), self.max_pool(graph, n_feat)], dim=-1)

        n_feat = F.relu(self.conv2(graph, n_feat, e_feat))
        graph, n_feat, e_feat, _ = self.pool2(graph, n_feat, e_feat)
        final_readout = final_readout + torch.cat([self.avg_pool(graph, n_feat), self.max_pool(graph, n_feat)], dim=-1)

        n_feat = F.relu(self.conv3(graph, n_feat, e_feat))
        final_readout = final_readout + torch.cat([self.avg_pool(graph, n_feat), self.max_pool(graph, n_feat)], dim=-1)

        n_feat = F.relu(self.lin1(n_feat))
        n_feat = F.dropout(n_feat, p=self.dropout, training=self.training)
        n_feat = F.relu(self.lin2(n_feat))
        n_feat = F.dropout(n_feat, p=self.dropout, training=self.training)
        n_feat = self.lin3(n_feat)

        return F.log_softmax(n_feat, dim=-1)
