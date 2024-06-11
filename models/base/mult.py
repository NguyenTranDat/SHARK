import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import ModelConfig, AudioConfig, VideoConfig, TextConfig
from libs.transformer import TransformerEncoder


class MULT(nn.Module):
    def __init__(self):
        super(MULT, self).__init__()

        self.embed_dropout = 0.25

        self.d_l = self.d_a = self.d_v = ModelConfig.MULT_DESTINATION_FEATURE_DIM

        self.proj_l = nn.Conv1d(
            TextConfig.MAX_SEQ_LEN,
            self.d_l,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        self.proj_a = nn.Conv1d(
            AudioConfig.MAX_SEQ_LEN,
            self.d_a,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        self.proj_v = nn.Conv1d(
            VideoConfig.MAX_SEQ_LEN,
            self.d_v,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        self.trans_l_with_a = self.get_network(self_type="la")
        self.trans_l_with_v = self.get_network(self_type="lv")
        self.trans_a_with_l = self.get_network(self_type="al")
        self.trans_a_with_v = self.get_network(self_type="av")
        self.trans_v_with_l = self.get_network(self_type="vl")
        self.trans_v_with_a = self.get_network(self_type="va")

        self.trans_l_mem = self.get_network(self_type="l_mem", layers=3)
        self.trans_a_mem = self.get_network(self_type="a_mem", layers=3)
        self.trans_v_mem = self.get_network(self_type="v_mem", layers=3)

    def get_network(self, self_type: str ="l", layers: int =-1):
        if self_type in ["l", "al", "vl"]:
            embed_dim, attn_dropout = (
                self.d_l,
                ModelConfig.MULT_ATTENTION_DROPOUT,
            )
        elif self_type in ["a", "la", "va"]:
            embed_dim, attn_dropout = (
                self.d_a,
                AudioConfig.MULT_ATTENTION_DROPOUT,
            )
        elif self_type in ["v", "lv", "av"]:
            embed_dim, attn_dropout = (
                self.d_v,
                VideoConfig.MULT_ATTENTION_DROPOUT,
            )
        elif self_type == "l_mem":
            embed_dim, attn_dropout = (
                2 * self.d_l,
                ModelConfig.MULT_ATTENTION_DROPOUT,
            )
        elif self_type == "a_mem":
            embed_dim, attn_dropout = (
                2 * self.d_a,
                ModelConfig.MULT_ATTENTION_DROPOUT,
            )
        elif self_type == "v_mem":
            embed_dim, attn_dropout = (
                2 * self.d_v,
                ModelConfig.MULT_ATTENTION_DROPOUT,
            )
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=ModelConfig.MULT_NUM_HEADS,
            layers=max(ModelConfig.MULT_NUM_LAYERS, layers),
            attn_dropout=attn_dropout,
            relu_dropout=ModelConfig.MULT_RELU_DROPOUT,
            res_dropout=ModelConfig.MULT_RES_DROPOUT,
            embed_dropout=self.embed_dropout,
            attn_mask=ModelConfig.MULT_ATTENTION_MASK,
        )

    def forward(self, text_feats, video_feats, audio_feats):
        text_feats = F.dropout(
            text_feats, p=self.embed_dropout, training=self.training
        )

        proj_x_l = self.proj_l(text_feats)
        proj_x_a = self.proj_a(audio_feats)
        proj_x_v = self.proj_v(video_feats)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)

        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)

        return h_ls, h_as, h_vs
