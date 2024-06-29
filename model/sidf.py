import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(dotenv_path)

SDIF_FEATURE_DIM = int(os.getenv("SDIF_FEATURE_DIM"))
SDIF_NUMLAYER_SELF_ATTENTION = int(os.getenv("SDIF_NUMLAYER_SELF_ATTENTION"))
SDIF_NUM_HEAD = int(os.getenv("SDIF_NUM_HEAD"))


class SDIF(nn.Module):
    def __init__(self):
        super(SDIF, self).__init__()
        self.mlp_project = nn.Sequential(nn.Linear(SDIF_FEATURE_DIM, SDIF_FEATURE_DIM), nn.Dropout(0.1), nn.GELU())

        encoder_layer = nn.TransformerEncoderLayer(d_model=SDIF_FEATURE_DIM, nhead=SDIF_NUM_HEAD)
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=SDIF_NUMLAYER_SELF_ATTENTION)

    def forward(self, all_reps, shallow_seq, text_mask, video_mask, audio_mask):
        # Third layer: mlp->VAL
        shallow_seq = self.mlp_project(shallow_seq)

        # Deep Interaction
        tri_cat_mask = torch.cat([text_mask, video_mask, audio_mask], dim=-1)

        tri_mask_len = torch.sum(tri_cat_mask, dim=1, keepdim=True)
        shallow_masked_output = torch.mul(tri_cat_mask.unsqueeze(2), shallow_seq)
        shallow_rep = torch.sum(shallow_masked_output, dim=1, keepdim=False) / tri_mask_len

        all_reps = torch.cat((all_reps, shallow_rep.unsqueeze(0)), dim=0)

        all_hiddens = self.self_attention(all_reps)
        deep_rep = torch.cat([all_hiddens[i] for i in range(7)], dim=1)

        return deep_rep
