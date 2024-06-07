import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_attention_layer import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, h, h_xReact, h_oReact, utt_xReact_mask, utt_oReact_mask):
        x = F.dropout(h, self.dropout, training=self.training)
        h_xReact = F.dropout(h_xReact, self.dropout, training=self.training)
        h_oReact = F.dropout(h_oReact, self.dropout, training=self.training)
        x = torch.cat(
            [att(x, h_xReact, h_oReact, utt_xReact_mask, utt_oReact_mask) for att in self.attentions],
            dim=-1,
        )  # Concatenate multiple different features obtained from multiple attention mechanisms on the same node to form a long feature.
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, h_xReact, h_oReact, utt_xReact_mask, utt_oReact_mask))
        return x


__all__ = ["GAT", "GraphAttentionLayer"]
