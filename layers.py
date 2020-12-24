import dgl
import torch
import dgl.function as fn


class NodeInfoScoreLayer(torch.nn.Module):
    def __init__(self, sym_norm:bool=True):
        super(NodeInfoScoreLayer, self).__init__()
        self.sym_norm = sym_norm

    def forward(self, graph:dgl.DGLGraph, feat:torch.Tensor):
        with graph.local_scope():
            if self.sym_norm:
                src_norm = torch.pow(graph.out_degrees().float().clamp(min=1), -0.5)
                src_norm = src_norm.view(-1, 1).to(feat.device)
                dst_norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
                dst_norm = dst_norm.view(-1, 1).to(feat.device)

                src_feat = feat * src_norm
                
                graph.ndata["h"] = src_feat
                graph.update_all(fn.copy_src("h", "m"), fn.sum("m", "h"))
                
                dst_feat = graph.ndata.pop("h") * dst_norm
                feat = feat - dst_feat
            else:
                dst_norm = 1. / graph.in_degrees().float().clamp(min=1)
                dst_norm = dst_norm.view(-1, 1)

                graph.ndata["h"] = feat
                graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))

                feat = feat - dst_norm * graph.ndata.pop("h")

            score = torch.sum(torch.abs(feat), dim=1)
            return score
