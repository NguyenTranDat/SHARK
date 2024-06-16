import torch
from transformers import BartConfig
from transformers.models.bart.modeling_bart import BartEncoder

from src.lib.Seq2Seq.encoder import Seq2SeqEncoder
from src.lib.ulti.attention import Attention
from src.lib.ulti import seq_len_to_mask


class FBartEncoder(Seq2SeqEncoder):
    def __init__(self, encoder):
        super().__init__()
        assert isinstance(encoder, BartEncoder)
        self.bart_encoder = encoder

    def forward(self, src_tokens, src_seq_len):
        mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        dict = self.bart_encoder(
            input_ids=src_tokens,
            attention_mask=mask,
            return_dict=True,
            output_hidden_states=True,
        )
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        return encoder_outputs, mask, hidden_states
