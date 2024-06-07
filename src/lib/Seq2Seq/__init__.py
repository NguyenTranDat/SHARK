import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import Seq2SeqDecoder
from .encoder import Seq2SeqEncoder


class Seq2SeqModel(nn.Module):
    def __init__(self, encoder: Seq2SeqEncoder, decoder: Seq2SeqDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        src_tokens: "torch.LongTensor",
        tgt_tokens: "torch.LongTensor",
        src_seq_len: "torch.LongTensor" = None,
        tgt_seq_len: "torch.LongTensor" = None,
    ):
        state = self.prepare_state(src_tokens, src_seq_len)
        decoder_output = self.decoder(tgt_tokens, state)
        if isinstance(decoder_output, torch.Tensor):
            return {"pred": decoder_output}
        elif isinstance(decoder_output, (tuple, list)):
            return {"pred": decoder_output[0]}
        else:
            raise TypeError(
                f"Unsupported return type from Decoder:{type(self.decoder)}"
            )

    def train_step(
        self,
        src_tokens: "torch.LongTensor",
        tgt_tokens: "torch.LongTensor",
        src_seq_len: "torch.LongTensor" = None,
        tgt_seq_len: "torch.LongTensor" = None,
    ):
        res = self(src_tokens, tgt_tokens, src_seq_len, tgt_seq_len)
        pred = res["pred"]
        if tgt_seq_len is not None:

            batch_size = tgt_seq_len.shape[0]
            broad_cast_seq_len = (
                torch.arange(tgt_tokens.size(1)).expand(batch_size, -1).to(tgt_seq_len)
            )
            mask = broad_cast_seq_len < tgt_seq_len.unsqueeze(1)
            tgt_tokens = tgt_tokens.masked_fill(mask.eq(0), -100)
        loss = F.cross_entropy(pred[:, :-1].transpose(1, 2), tgt_tokens[:, 1:])
        return {"loss": loss}

    def prepare_state(
        self, src_tokens: "torch.LongTensor", src_seq_len: "torch.LongTensor" = None
    ):
        encoder_output, encoder_mask = self.encoder(src_tokens, src_seq_len)
        state = self.decoder.init_state(encoder_output, encoder_mask)
        return state

    @classmethod
    def build_model(cls, *args, **kwargs):
        raise NotImplementedError(
            "A `Seq2SeqModel` must implement its own classmethod `build_model()`."
        )
