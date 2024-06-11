import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import ModelConfig, AudioConfig, VideoConfig, TextConfig
from libs.transformer import TransformerEncoder


class SDIF(nn.Module):
    def __init__(self):
        super(SDIF, self).__init__()
        self.mlp_project = nn.Sequential(
            nn.Linear(
                ModelConfig.SDIF_FEATURE_DIM, ModelConfig.SDIF_FEATURE_DIM
            ),
            nn.Dropout(ModelConfig.SDIF_DROPOUT),
            nn.GELU(),
        )

        self.self_att = TransformerEncoder(
            embed_dim=ModelConfig.SDIF_FEATURE_DIM,
            num_heads=ModelConfig.SDIF_NUM_HEAD,
            layers=ModelConfig.SDIF_NUMLAYER_SELF_ATTENTION,
            attn_mask=ModelConfig.MULT_ATTENTION_MASK,
        )


    def forward(
        self,
        text_rep,
        video_rep,
        audio_rep,
        shallow_seq,
        mult_t_rep,
        mult_a_rep,
        mult_v_rep,
        text_mask,
        video_mask,
        audio_mask,
    ):
        # Third layer: mlp->VAL
        shallow_seq = self.mlp_project(shallow_seq)

        # Deep Interaction
        tri_cat_mask = torch.cat([text_mask, video_mask, audio_mask], dim=-1)

        tri_mask_len = torch.sum(tri_cat_mask, dim=1, keepdim=True)
        shallow_masked_output = torch.mul(
            tri_cat_mask.unsqueeze(2), shallow_seq
        )
        shallow_rep = (
            torch.sum(shallow_masked_output, dim=1, keepdim=False)
            / tri_mask_len
        )

        all_reps = torch.stack(
            (
                text_rep,
                video_rep,
                audio_rep,
                mult_t_rep,
                mult_a_rep,
                mult_v_rep,
                shallow_rep,
            ),
            dim=0,
        )

        all_hiddens = self.self_att(all_reps)
        deep_rep = torch.cat([all_hiddens[i] for i in range(7)], dim=1)

        return deep_rep
