import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartModel, BartTokenizer, AutoTokenizer

from src.lib.Seq2Seq import Seq2SeqModel
from src.lib.State.bart import BartState
from src.lib.GAT import GAT, GraphAttentionLayer
from src.lib.BART.classification_head import BartClassificationHead
from src.lib.BART.encoder import FBartEncoder
from src.lib.BART.ulti import get_utt_representation


class BartSeq2SeqModel(nn.Module):
    def __init__(self):
        """
        Define the encoder and decoder
        Initialize the custom tokens
        """
        super(BartSeq2SeqModel, self).__init__()
        self.hidden_size = 768

        tokenizer = AutoTokenizer.from_pretrained("src/data/tokenizer")

        model = BartModel.from_pretrained("facebook/bart-base")
        num_tokens, _ = model.encoder.embed_tokens.weight.shape
        model.resize_token_embeddings(len(tokenizer) + num_tokens)
        encoder = model.encoder
        decoder = model.decoder
        _tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        vocab = tokenizer.get_vocab()
        for token, token_id in vocab.items():
            if token[:2] == "<<" and len(token) != 2:
                index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                if len(index) > 1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]
                assert index >= num_tokens, (index, num_tokens, token)
                indexes = _tokenizer.convert_tokens_to_ids(
                    _tokenizer.tokenize(token[2:-2])
                )
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                model.decoder.embed_tokens.weight.data[index] = embed
        self.encoder = FBartEncoder(encoder)
        self.linear_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 3, 2), nn.Sigmoid()
        )
        self.graph_att_layer = GraphAttentionLayer(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            dropout=0.2,
            alpha=0.2,
        )
        self.emo_ffn = BartClassificationHead(
            inner_dim=self.hidden_size,
            input_dim=self.hidden_size,
            num_classes=30,
            pooler_dropout=0.3,
        )

    def forward(
        self,
        token,
        word_atomic_xReact,
        word_atomic_oReact,
        atomic_prefix_ids_xReact,
        atomic_prefix_ids_oReact,
        word_retrieval_xReact,
        word_retrieval_oReact,
        retrieval_prefix_ids_xReact,
        retrieval_prefix_ids_oReact,
        utt_prefix_ids,
        len_word_atomic_oReact,
        len_word_atomic_xReact,
        len_word_retrieval_xReact,
        len_word_retrieval_oReact,
        utt_xReact_mask,
        utt_oReact_mask,
        dia_utt_num,
        len_token,
    ):

        encoder_outputs, encoder_mask, hidden_states = self.encoder(token, len_token)
        src_embed_outputs = hidden_states[0]

        encoder_outputs_utt = get_utt_representation(
            encoder_outputs,
            utt_prefix_ids,
            dia_utt_num,
        )
        encoder_outputs_xReact, encoder_mask_xReact, hidden_states_xReact = (
            self.encoder(word_atomic_xReact, len_word_atomic_xReact)
        )
        encoder_outputs_oReact, encoder_mask_oReact, hidden_states_oReact = (
            self.encoder(word_atomic_oReact, len_word_atomic_oReact)
        )
        encoder_outputs_utt_xReact = get_utt_representation(
            encoder_outputs_xReact,
            atomic_prefix_ids_xReact,
            dia_utt_num,
        )
        encoder_outputs_utt_oReact = get_utt_representation(
            encoder_outputs_oReact,
            atomic_prefix_ids_oReact,
            dia_utt_num,
        )

        (
            encoder_outputs_xReact_retrieval,
            encoder_mask_xReact_retrieval,
            hidden_states_xReact_retrieval,
        ) = self.encoder(word_retrieval_xReact, len_word_retrieval_xReact)
        (
            encoder_outputs_oReact_retrieval,
            encoder_mask_oReact_retrieval,
            hidden_states_oReact_retrieval,
        ) = self.encoder(word_retrieval_oReact, len_word_retrieval_oReact)

        encoder_outputs_utt_xReact_retrieval = get_utt_representation(
            encoder_outputs_xReact_retrieval,
            retrieval_prefix_ids_xReact,
            dia_utt_num,
        )
        encoder_outputs_utt_oReact_retrieval = get_utt_representation(
            encoder_outputs_oReact_retrieval,
            retrieval_prefix_ids_oReact,
            dia_utt_num,
        )

        xReact_encoder_outputs_utt = torch.cat(
            (
                encoder_outputs_utt,
                encoder_outputs_utt_xReact,
                encoder_outputs_utt_xReact_retrieval,
            ),
            -1,
        )

        oReact_encoder_outputs_utt = torch.cat(
            (
                encoder_outputs_utt,
                encoder_outputs_utt_oReact,
                encoder_outputs_utt_oReact_retrieval,
            ),
            -1,
        )

        indicator_x = self.linear_layer(xReact_encoder_outputs_utt)
        indicator_o = self.linear_layer(oReact_encoder_outputs_utt)
        indicator_x_ = F.softmax(indicator_x, dim=-1)
        indicator_o_ = F.softmax(indicator_o, dim=-1)

        indicator_x_ = (
            indicator_x_[:, :, 0]
            .unsqueeze(2)
            .repeat(1, 1, encoder_outputs_utt.size(-1))
        )

        indicator_o_ = (
            indicator_o_[:, :, 0]
            .unsqueeze(2)
            .repeat(1, 1, encoder_outputs_utt.size(-1))
        )

        new_xReact_encoder_outputs_utt = (
            indicator_x_ * encoder_outputs_utt_xReact
            + (1 - indicator_x_) * encoder_outputs_utt_xReact_retrieval
        )

        new_oReact_encoder_outputs_utt = (
            indicator_o_ * encoder_outputs_utt_oReact
            + (1 - indicator_o_) * encoder_outputs_utt_oReact_retrieval
        )

        new_encoder_outputs_utt = self.graph_att_layer(
            encoder_outputs_utt,
            new_xReact_encoder_outputs_utt,
            new_oReact_encoder_outputs_utt,
            utt_xReact_mask,
            utt_oReact_mask,
        )

        batch_size = dia_utt_num.shape[0]

        broad_cast_seq_len = (
            torch.arange(encoder_outputs_utt.size(1))
            .expand(batch_size, -1)
            .to(dia_utt_num)
        )

        utt_mask = broad_cast_seq_len < dia_utt_num.unsqueeze(1)

        new_encoder_outputs_utt = new_encoder_outputs_utt.masked_fill(
            utt_mask.eq(0).unsqueeze(2).repeat(1, 1, encoder_outputs_utt.size(-1)),
            0,
        )

        logits = self.emo_ffn(new_encoder_outputs_utt)

        logits = logits.view(1, -1)[:, :30]

        logits = F.softmax(logits, dim=-1)

        # bz, _, _ = new_encoder_outputs_utt.size()
        # new_encoder_outputs = encoder_outputs.clone()

        # for ii in range(bz):
        #     for jj in range(dia_utt_num[ii]):
        #         new_encoder_outputs[ii, utt_prefix_ids[ii][jj]] = (
        #             encoder_outputs[ii, utt_prefix_ids[ii][jj]]
        #             + new_encoder_outputs_utt[ii, jj]
        #         )  # Add the origin prefix representation to the knowledge-enhanced utterance representation

        # state = BartState(
        #     new_encoder_outputs, encoder_mask, src_tokens, first, src_embed_outputs
        # )
        # return state, logits

        return logits
