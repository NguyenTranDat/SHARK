import torch
import torch.nn as nn
import torch.nn.functional as F

from src.lib.Seq2Seq.decoder import Seq2SeqDecoder


class CaGFBartDecoder(Seq2SeqDecoder):
    # Copy and generate
    def __init__(self, decoder, pad_token_id, label_ids):
        super().__init__(decoder, pad_token_id, label_ids)
        assert isinstance(decoder, BartDecoder)
        self.decoder = decoder
        causal_mask = torch.zeros(512, 512).fill_(float("-inf"))
        causal_mask = causal_mask.triu(diagonal=1)
        self.register_buffer("causal_masks", causal_mask.float())
        self.pad_token_id = pad_token_id
        self.label_start_id = label_ids[0]
        self.label_end_id = label_ids[-1]
        mapping = torch.LongTensor([0, 2] + label_ids)
        self.register_buffer("mapping", mapping)
        self.src_start_index = len(mapping) - 1
        hidden_size = decoder.embed_tokens.weight.size(1)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, tokens, utt_prefix_ids, dia_utt_num, state):
        if tokens.size(0) != utt_prefix_ids.size(0):
            beam_size = tokens.size(0) // utt_prefix_ids.size(0)
            utt_prefix_ids = utt_prefix_ids.repeat_interleave(beam_size, dim=0)
            dia_utt_num = dia_utt_num.repeat_interleave(beam_size, dim=0)

        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask
        tgt_tokens_copy = tokens

        first = state.first

        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # mapping to the BART token index
        mapping_token_mask = tokens.lt(self.src_start_index)
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index  # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)

        tokens = torch.where(
            mapping_token_mask, tag_mapped_tokens, word_mapped_tokens
        )  # bsz x max_len
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)

        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)
            dict = self.decoder(
                input_ids=tokens,
                encoder_hidden_states=encoder_outputs,
                encoder_padding_mask=encoder_pad_mask,
                decoder_padding_mask=decoder_pad_mask,
                decoder_causal_mask=self.causal_masks[
                    : tokens.size(1), : tokens.size(1)
                ],
                return_dict=True,
            )
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(
                input_ids=tokens,
                encoder_hidden_states=encoder_outputs,
                encoder_padding_mask=encoder_pad_mask,
                decoder_padding_mask=None,
                decoder_causal_mask=None,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size.
        hidden_state = self.dropout_layer(hidden_state)
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full(
            (
                hidden_state.size(0),
                hidden_state.size(1),
                self.src_start_index + src_tokens.size(-1),
            ),
            fill_value=-1e24,
        )  # bsz x max_len x (max_word_len+2+num_labels)

        eos_scores = F.linear(
            hidden_state, self.dropout_layer(self.decoder.embed_tokens.weight[2:3])
        )  # bsz x max_len x 1
        tag_scores = F.linear(
            hidden_state,
            self.dropout_layer(
                self.decoder.embed_tokens.weight[
                    self.label_start_id : self.label_end_id
                ]
            ),
        )  # bsz x max_len x num_class

        src_outputs = state.encoder_output
        src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len
            src_outputs = src_outputs.gather(
                index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1
            )
        else:
            mask = state.encoder_mask.eq(0)

        mask = mask.unsqueeze(1)
        input_embed = self.dropout_layer(
            self.decoder.embed_tokens(src_tokens)
        )  # bsz x max_word_len x hidden_size

        src_outputs = (src_outputs + input_embed) / 2

        word_scores = torch.einsum(
            "blh,bnh->bln", hidden_state, src_outputs
        )  # bsz x max_len x max_word_len
        mask = mask.__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2 : self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index :] = word_scores

        return logits

    def decode(self, tokens, utt_prefix_ids, dia_utt_num, state):
        return self(tokens, utt_prefix_ids, dia_utt_num, state)[:, -1]
