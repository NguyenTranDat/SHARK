import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartModel, BartTokenizer, AutoTokenizer, BertModel

from src.lib.Seq2Seq import Seq2SeqModel
from src.lib.State.bart import BartState
from src.lib.GAT import GAT, GraphAttentionLayer
from src.lib.BART.classification_head import BartClassificationHead
from src.lib.BART.encoder import FBartEncoder
from src.lib.BART.decoder import CaGFBartDecoder
from src.lib.ulti import get_utt_representation, seq_len_to_mask


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BartSeq2SeqModel(nn.Module):
    def __init__(self):
        """
        Define the encoder and decoder
        Initialize the custom tokens
        """
        super(BartSeq2SeqModel, self).__init__()
        self.hidden_size = 768

        tokenizer = AutoTokenizer.from_pretrained("src/example/log/tokenizer")
        self.encoder = BertModel.from_pretrained("bert-base-cased")
        self.encoder.resize_token_embeddings(len(tokenizer))
        self.linear_layer = nn.Sequential(nn.Linear(self.hidden_size * 3, 2), nn.ReLU())
        self.graph_att_layer = GraphAttentionLayer(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            dropout=0.1,
            alpha=0.2,
        )
        self.emo_ffn = BartClassificationHead(
            inner_dim=self.hidden_size,
            input_dim=self.hidden_size,
            num_classes=30,
            pooler_dropout=0.2,
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
        encoder_outputs = self.encoder(
            token[:, 0].to(device), token[:, 1].to(device), token[:, 2].to(device)
        ).last_hidden_state
        encoder_outputs_xReact = self.encoder(
            word_atomic_xReact[:, 0].to(device),
            word_atomic_xReact[:, 1].to(device),
            word_atomic_xReact[:, 2].to(device),
        ).last_hidden_state
        encoder_outputs_oReact = self.encoder(
            word_atomic_oReact[:, 0].to(device),
            word_atomic_oReact[:, 1].to(device),
            word_atomic_oReact[:, 2].to(device),
        ).last_hidden_state
        encoder_outputs_xReact_retrieval = self.encoder(
            word_retrieval_xReact[:, 0].to(device),
            word_retrieval_xReact[:, 1].to(device),
            word_retrieval_xReact[:, 2].to(device),
        ).last_hidden_state
        encoder_outputs_oReact_retrieval = self.encoder(
            word_retrieval_oReact[:, 0].to(device),
            word_retrieval_oReact[:, 1].to(device),
            word_retrieval_oReact[:, 2].to(device),
        ).last_hidden_state

        encoder_outputs_utt = get_utt_representation(
            encoder_outputs,
            utt_prefix_ids.to(device),
            dia_utt_num.to(device),
        )

        encoder_outputs_utt_xReact = get_utt_representation(
            encoder_outputs_xReact,
            atomic_prefix_ids_xReact.to(device),
            dia_utt_num.to(device),
        )

        encoder_outputs_utt_oReact = get_utt_representation(
            encoder_outputs_oReact,
            atomic_prefix_ids_oReact.to(device),
            dia_utt_num.to(device),
        )

        encoder_outputs_utt_xReact_retrieval = get_utt_representation(
            encoder_outputs_xReact_retrieval,
            retrieval_prefix_ids_xReact.to(device),
            dia_utt_num.to(device),
        )
        encoder_outputs_utt_oReact_retrieval = get_utt_representation(
            encoder_outputs_oReact_retrieval,
            retrieval_prefix_ids_oReact.to(device),
            dia_utt_num.to(device),
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

        indicator_x_ = indicator_x_[:, :, 0].unsqueeze(2).repeat(1, 1, encoder_outputs_utt.size(-1))

        indicator_o_ = indicator_o_[:, :, 0].unsqueeze(2).repeat(1, 1, encoder_outputs_utt.size(-1))

        new_xReact_encoder_outputs_utt = (
            indicator_x_ * encoder_outputs_utt_xReact + (1 - indicator_x_) * encoder_outputs_utt_xReact_retrieval
        )

        new_oReact_encoder_outputs_utt = (
            indicator_o_ * encoder_outputs_utt_oReact + (1 - indicator_o_) * encoder_outputs_utt_oReact_retrieval
        )

        new_encoder_outputs_utt = self.graph_att_layer(
            encoder_outputs_utt,
            new_xReact_encoder_outputs_utt,
            new_oReact_encoder_outputs_utt,
            utt_xReact_mask.to(device),
            utt_oReact_mask.to(device),
        )

        utt_mask = seq_len_to_mask(dia_utt_num.to(device), max_len=encoder_outputs_utt.size(1))  # bsz x max_utt_len

        new_encoder_outputs_utt = new_encoder_outputs_utt.masked_fill(
            utt_mask.eq(0).unsqueeze(2).repeat(1, 1, encoder_outputs_utt.size(-1)),
            0,
        )

        logits = self.emo_ffn(new_encoder_outputs_utt)

        return logits
