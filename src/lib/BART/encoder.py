import torch
from transformers import BartConfig
from transformers.models.bart.modeling_bart import BartEncoder

from src.lib.Seq2Seq.encoder import Seq2SeqEncoder
from src.lib.ulti.attention import Attention


# class EncoderLayer(nn.Module):
#     def __init__(self, config: BartConfig):
#         super().__init__()
#         self.embed_dim = config.d_model
#         self.self_attn = Attention(
#             self.embed_dim,
#             config.encoder_attention_heads,
#             dropout=config.attention_dropout,
#         )
#         self.normalize_before = config.normalize_before
#         self.self_attn_layer_norm = LayerNorm(self.embed_dim)
#         self.dropout = config.dropout
#         self.activation_fn = GELUActivation()
#         self.activation_dropout = config.activation_dropout
#         self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
#         self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
#         self.final_layer_norm = LayerNorm(self.embed_dim)

#     def forward(self, x, encoder_padding_mask, output_attentions=False):
#         """
#         Args:
#             x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
#             encoder_padding_mask (ByteTensor): binary ByteTensor of shape
#                 `(batch, src_len)` where padding elements are indicated by ``1``.
#             for t_tgt, t_src is excluded (or masked out), =0 means it is
#             included in attention

#         Returns:
#             encoded output of shape `(seq_len, batch, embed_dim)`
#         """
#         residual = x
#         if self.normalize_before:
#             x = self.self_attn_layer_norm(x)
#         x, attn_weights = self.self_attn(
#             query=x,
#             key=x,
#             key_padding_mask=encoder_padding_mask,
#             output_attentions=output_attentions,
#         )
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = residual + x
#         if not self.normalize_before:
#             x = self.self_attn_layer_norm(x)

#         residual = x
#         if self.normalize_before:
#             x = self.final_layer_norm(x)
#         x = self.activation_fn(self.fc1(x))
#         x = F.dropout(x, p=self.activation_dropout, training=self.training)
#         x = self.fc2(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = residual + x
#         if not self.normalize_before:
#             x = self.final_layer_norm(x)
#         if torch.isinf(x).any() or torch.isnan(x).any():
#             clamp_value = torch.finfo(x.dtype).max - 1000
#             x = torch.clamp(x, min=-clamp_value, max=clamp_value)
#         return x, attn_weights


# class BartEncoder(nn.Module):
#     """
#     Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
#     is a :class:`EncoderLayer`.

#     Args:
#         config: BartConfig
#     """

#     def __init__(self, config: BartConfig, embed_tokens):
#         super().__init__()

#         self.dropout = config.dropout
#         self.layerdrop = config.encoder_layerdrop

#         embed_dim = embed_tokens.embedding_dim
#         self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
#         self.padding_idx = embed_tokens.padding_idx
#         self.max_source_positions = config.max_position_embeddings

#         self.embed_tokens = embed_tokens
#         self.embed_positions = SinusoidalPositionalEmbedding(
#             config.max_position_embeddings, embed_dim, self.padding_idx
#         )
#         self.layers = nn.ModuleList(
#             [EncoderLayer(config) for _ in range(config.encoder_layers)]
#         )
#         self.layernorm_embedding = (
#             LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
#         )
#         # mbart has one extra layer_norm
#         self.layer_norm = (
#             LayerNorm(config.d_model) if config.add_final_layer_norm else None
#         )

#     def forward(
#         self,
#         input_ids,
#         attention_mask=None,
#         output_attentions=False,
#         output_hidden_states=False,
#         return_dict=False,
#     ):
#         """
#         Args:
#             input_ids (LongTensor): tokens in the source language of shape
#                 `(batch, src_len)`
#             attention_mask (torch.LongTensor): indicating which indices are padding tokens.
#         Returns:
#             BaseModelOutput or Tuple comprised of:
#                 - **x** (Tensor): the last encoder layer's output of
#                   shape `(src_len, batch, embed_dim)`
#                 - **encoder_states** (tuple(torch.FloatTensor)): all intermediate
#                   hidden states of shape `(src_len, batch, embed_dim)`.
#                   Only populated if *output_hidden_states:* is True.
#                 - **all_attentions** (tuple(torch.FloatTensor)): Attention weights for each layer.
#                 During training might not be of length n_layers because of layer dropout.
#         """
#         # check attention mask and invert
#         if attention_mask is not None:
#             attention_mask = invert_mask(attention_mask)

#         inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
#         embed_pos = self.embed_positions(input_ids)
#         x = inputs_embeds + embed_pos
#         x = self.layernorm_embedding(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         # B x T x C -> T x B x C
#         x = x.transpose(0, 1)

#         encoder_states = [] if output_hidden_states else None
#         all_attentions = () if output_attentions else None
#         for encoder_layer in self.layers:
#             if output_hidden_states:
#                 encoder_states.append(x)
#             dropout_probability = random.uniform(0, 1)
#             if self.training and (
#                 dropout_probability < self.layerdrop
#             ):  # skip the layer
#                 attn = None
#             else:
#                 x, attn = encoder_layer(
#                     x, attention_mask, output_attentions=output_attentions
#                 )

#             if output_attentions:
#                 all_attentions = all_attentions + (attn,)

#         if self.layer_norm:
#             x = self.layer_norm(x)
#         if output_hidden_states:
#             encoder_states.append(x)
#             # T x B x C -> B x T x C
#             encoder_states = tuple(
#                 hidden_state.transpose(0, 1) for hidden_state in encoder_states
#             )

#         # T x B x C -> B x T x C
#         x = x.transpose(0, 1)

#         if not return_dict:
#             return tuple(
#                 v for v in [x, encoder_states, all_attentions] if v is not None
#             )
#         return BaseModelOutput(
#             last_hidden_state=x, hidden_states=encoder_states, attentions=all_attentions
#         )


class FBartEncoder(Seq2SeqEncoder):
    def __init__(self, encoder):
        super().__init__()
        assert isinstance(encoder, BartEncoder)
        self.bart_encoder = encoder

    def forward(self, src_tokens, src_seq_len):
        batch_size = src_seq_len.shape[0]
        broad_cast_seq_len = (
            torch.arange(src_tokens.size(1)).expand(batch_size, -1).to(src_seq_len)
        )
        mask = broad_cast_seq_len < src_seq_len.unsqueeze(1)
        dict = self.bart_encoder(
            input_ids=src_tokens,
            attention_mask=mask,
            return_dict=True,
            output_hidden_states=True,
        )
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        return encoder_outputs, mask, hidden_states
