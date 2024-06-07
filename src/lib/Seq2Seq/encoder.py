import torch
import torch.nn as nn


class Seq2SeqEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tokens: torch.LongTensor, seq_len: torch.LongTensor):
        raise NotImplementedError
