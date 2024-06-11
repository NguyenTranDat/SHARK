import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import ModelConfig, AudioConfig, VideoConfig, TextConfig
from libs.bert import BERTEncoder

from .mult import MULT
from .sdif import SDIF


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = torch.device("cpu")

        self.text_subnet = BERTEncoder.from_pretrained(
            TextConfig.TEXT_BACKBONE
        )

        self.v2t_project = nn.Linear(
            VideoConfig.HIDDEN_DIM, TextConfig.HIDDEN_DIM
        )

        self.mult_layer = MULT().to(self.device)

        self.mult_t_project = nn.Linear(
            ModelConfig.MULT_DESTINATION_FEATURE_DIM * 2,
            TextConfig.MAX_SEQ_LEN,
        )
        self.mult_v_project = nn.Linear(
            ModelConfig.MULT_DESTINATION_FEATURE_DIM * 2,
            VideoConfig.MAX_SEQ_LEN,
        )
        self.mult_a_project = nn.Linear(
            ModelConfig.MULT_DESTINATION_FEATURE_DIM * 2,
            AudioConfig.MAX_SEQ_LEN,
        )

        self.sdif_layer = SDIF().to(self.device)

        self.fusion = nn.Sequential(
            nn.Linear(
                ModelConfig.SDIF_FEATURE_DIM * 7,
                ModelConfig.SDIF_FEATURE_DIM * 3,
            ),
            nn.Dropout(ModelConfig.SDIF_DROPOUT),
            nn.GELU(),
            nn.Linear(
                ModelConfig.SDIF_FEATURE_DIM * 3, ModelConfig.num_labels
            ),
        )

        if ModelConfig.SDIF_AUGMENT_TEXT:
            self.aug_dp = nn.Dropout(ModelConfig.SDIF_DROPOUT)
            self.out_layer = nn.Linear(
                ModelConfig.SDIF_FEATURE_DIM, ModelConfig.num_labels
            )

    def forward(self, text_feats, video_feats, audio_feats):
        # first layer : T,V,A
        video_seq = self.v2t_project(video_feats.float().to(self.device))
        audio_seq = audio_feats.float().to(self.device)
        bert_sent_mask = text_feats[:, 1].float().to(self.device)
        text_outputs = self.text_subnet(text_feats)
        text_seq, text_rep = (
            text_outputs["last_hidden_state"],
            text_outputs["pooler_output"],
        )

        video_mask = (
            torch.sum(
                video_feats.ne(
                    torch.zeros(video_feats[0].shape[-1]).to(self.device)
                ).int(),
                dim=-1,
            )
            / video_feats[0].shape[-1]
        )
        video_mask_len = torch.sum(video_mask, dim=1, keepdim=True)
        video_mask_len = torch.where(
            video_mask_len > 0.5,
            video_mask_len,
            torch.ones([1]).to(self.device),
        )
        video_masked_output = torch.mul(video_mask.unsqueeze(2), video_seq)
        video_rep = (
            torch.sum(video_masked_output, dim=1, keepdim=False)
            / video_mask_len
        )

        audio_mask = (
            torch.sum(
                audio_feats.ne(
                    torch.zeros(audio_feats[0].shape[-1]).to(self.device)
                ).int(),
                dim=-1,
            )
            / audio_feats[0].shape[-1]
        )
        audio_mask_len = torch.sum(audio_mask, dim=1, keepdim=True)
        audio_masked_output = torch.mul(audio_mask.unsqueeze(2), audio_seq)
        audio_rep = (
            torch.sum(audio_masked_output, dim=1, keepdim=False)
            / audio_mask_len
        )

        mult_t_seq, mult_a_seq, mult_v_seq = self.mult_layer(
            text_feats=text_seq,
            video_feats=video_seq,
            audio_feats=audio_seq,
        )

        proj_a = self.mult_a_project(mult_a_seq)
        proj_a = proj_a.permute(1, 2, 0)
        mult_a_masked_output = torch.mul(audio_mask.unsqueeze(2), proj_a)
        mult_a_rep = (
            torch.sum(mult_a_masked_output, dim=1, keepdim=False)
            / audio_mask_len
        )

        proj_v = self.mult_v_project(mult_v_seq)
        proj_v = proj_v.permute(1, 2, 0)
        mult_v_masked_output = torch.mul(video_mask.unsqueeze(2), proj_v)
        mult_v_rep = (
            torch.sum(mult_v_masked_output, dim=1, keepdim=False)
            / video_mask_len
        )

        text_mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)
        proj_t = self.mult_t_project(mult_t_seq)
        proj_t = proj_t.permute(1, 2, 0)
        mult_t_masked_output = torch.mul(bert_sent_mask.unsqueeze(2), proj_t)
        mult_t_rep = (
            torch.sum(mult_t_masked_output, dim=1, keepdim=False)
            / text_mask_len
        )

        shallow_seq = torch.cat([proj_a, proj_t, proj_v], dim=1)

        sdif_kwargs = {
            "shallow_seq": shallow_seq,
            "text_rep": text_rep,
            "video_rep": video_rep,
            "audio_rep": audio_rep,
            "mult_t_rep": mult_t_rep,
            "mult_v_rep": mult_v_rep,
            "mult_a_rep": mult_a_rep,
            "text_mask": bert_sent_mask,
            "video_mask": video_mask,
            "audio_mask": audio_mask,
        }

        deep_rep = self.sdif_layer(**sdif_kwargs)

        logits = self.fusion(deep_rep)

        return logits

    def pre_train(self, text_feats):
        text_outputs = self.text_subnet(text_feats)
        text_seq, text_rep = (
            text_outputs["last_hidden_state"],
            text_outputs["pooler_output"],
        )
        text_rep = self.aug_dp(text_rep)
        logits = self.out_layer(text_rep)
        return logits
