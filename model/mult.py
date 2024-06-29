import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from dotenv import load_dotenv

from lib.transformer.encoder import TransformerEncoder
from process_data.benchmarks import benchmarks

dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(dotenv_path)

DATA_VERSION = os.getenv("DATA_VERSION")
MULT_DIM_MODEL = int(os.getenv("MULT_DIM_MODEL"))
MULT_NUM_HEAD = int(os.getenv("MULT_NUM_HEAD"))
MULT_NUM_LAYER = int(os.getenv("MULT_NUM_LAYER"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
benchmark = benchmarks[DATA_VERSION]


class MULT(nn.Module):
    def __init__(self):
        super(MULT, self).__init__()

        self.embed_dropout = 0.25
        self.d_l, self.d_a, self.d_v = MULT_DIM_MODEL, MULT_DIM_MODEL, MULT_DIM_MODEL

        self.max_seq_alignment_text = nn.Conv1d(
            benchmark["max_seq_lengths"]["text"], self.d_l, kernel_size=5, padding=0, bias=False
        )
        self.max_seq_alignment_audio = nn.Conv1d(
            benchmark["max_seq_lengths"]["audio"], self.d_a, kernel_size=1, padding=0, bias=False
        )
        self.max_seq_alignment_video = nn.Conv1d(
            benchmark["max_seq_lengths"]["video"], self.d_v, kernel_size=1, padding=0, bias=False
        )

        self.transfer_text_with_audio = self.get_network(self_type="la")
        self.transfer_text_with_video = self.get_network(self_type="lv")
        self.transfer_audio_with_text = self.get_network(self_type="al")
        self.transfer_audio_with_video = self.get_network(self_type="av")
        self.transfer_video_with_text = self.get_network(self_type="vl")
        self.transfer_video_with_audio = self.get_network(self_type="va")

        self.transfer_text_mem = self.get_network(self_type="l_mem", layers=3)
        self.transfer_audio_mem = self.get_network(self_type="a_mem", layers=3)
        self.transfer_video_mem = self.get_network(self_type="v_mem", layers=3)

    def get_network(self, self_type: str, layers: int = -1):
        if self_type in ["l", "al", "vl"]:
            d_model, dropout = self.d_l, 0.1
        elif self_type in ["a", "la", "va"]:
            d_model, dropout = self.d_a, 0.0
        elif self_type in ["v", "lv", "av"]:
            d_model, dropout = self.d_v, 0.0
        elif self_type == "l_mem":
            d_model, dropout = 2 * self.d_l, 0.1
        elif self_type == "a_mem":
            d_model, dropout = 2 * self.d_a, 0.1
        elif self_type == "v_mem":
            d_model, dropout = 2 * self.d_v, 0.1
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=d_model, num_heads=MULT_NUM_HEAD, layers=max(MULT_NUM_LAYER, layers))

    def forward(self, text_feats, video_feats, audio_feats):
        text_feats = F.dropout(text_feats, p=self.embed_dropout, training=self.training)

        text_alignment_seq = text_feats if text_feats.shape[-2] == self.d_l else self.max_seq_alignment_text(text_feats)
        video_alignment_seq = (
            video_feats if video_feats.shape[-2] == self.d_v else self.max_seq_alignment_video(video_feats)
        )
        audio_alignment_seq = (
            audio_feats if audio_feats.shape[-2] == self.d_l else self.max_seq_alignment_audio(audio_feats)
        )

        audio_alignment_seq = audio_alignment_seq.permute(2, 0, 1)
        video_alignment_seq = video_alignment_seq.permute(2, 0, 1)
        text_alignment_seq = text_alignment_seq.permute(2, 0, 1)

        # (V,A) --> L
        text_with_audio = self.transfer_text_with_audio(text_alignment_seq, audio_alignment_seq, audio_alignment_seq)
        text_with_video = self.transfer_text_with_video(text_alignment_seq, video_alignment_seq, video_alignment_seq)
        text_with_audio_video = torch.cat([text_with_audio, text_with_video], dim=2)
        text_with_audio_video = self.transfer_text_mem(text_with_audio_video)

        # (L,V) --> A
        audio_with_text = self.transfer_audio_with_text(audio_alignment_seq, text_alignment_seq, text_alignment_seq)
        audio_with_video = self.transfer_audio_with_video(audio_alignment_seq, video_alignment_seq, video_alignment_seq)
        audio_with_text_video = torch.cat([audio_with_text, audio_with_video], dim=2)
        audio_with_text_video = self.transfer_audio_mem(audio_with_text_video)

        # (L,A) --> V
        video_with_text = self.transfer_video_with_text(video_alignment_seq, text_alignment_seq, text_alignment_seq)
        video_with_audio = self.transfer_video_with_audio(video_alignment_seq, audio_alignment_seq, audio_alignment_seq)
        video_with_text_audio = torch.cat([video_with_text, video_with_audio], dim=2)
        video_with_text_audio = self.transfer_video_mem(video_with_text_audio)

        return text_with_audio_video, audio_with_text_video, video_with_text_audio
