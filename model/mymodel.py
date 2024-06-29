import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import BertModel

from process_data.benchmarks import benchmarks
from model.mult import MULT
from model.sidf import SDIF

dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(dotenv_path)

DATA_VERSION = os.getenv("DATA_VERSION")
TOKENIZER = os.getenv("TOKENIZER")
MULT_DIM_MODEL = int(os.getenv("MULT_DIM_MODEL"))
SDIF_FEATURE_DIM = int(os.getenv("SDIF_FEATURE_DIM"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
benchmark = benchmarks[DATA_VERSION]


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.hidden_dim = 768

        self.bert = BertModel.from_pretrained(TOKENIZER)

        self.hidden_dim_alignment_text = nn.Linear(benchmark["feat_dims"]["text"][TOKENIZER], self.hidden_dim)
        self.hidden_dim_alignment_audio = nn.Linear(benchmark["feat_dims"]["audio"], self.hidden_dim)
        self.hidden_dim_alignment_video = nn.Linear(benchmark["feat_dims"]["video"], self.hidden_dim)

        self.mult_layer = MULT().to(device)

        self.mult_text = nn.Linear(MULT_DIM_MODEL * 2, benchmark["max_seq_lengths"]["text"])
        self.mult_video = nn.Linear(MULT_DIM_MODEL * 2, benchmark["max_seq_lengths"]["video"])
        self.mult_audio = nn.Linear(MULT_DIM_MODEL * 2, benchmark["max_seq_lengths"]["audio"])
        self.sdif_layer = SDIF().to(device)

        self.fusion = nn.Sequential(
            nn.Linear(SDIF_FEATURE_DIM * 7, SDIF_FEATURE_DIM * 3),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(SDIF_FEATURE_DIM * 3, len(benchmark["intent_labels"])),
        )

    def forward(self, text, audio, video):
        bert_sent_mask = text[:, 1].float().to(device)
        text_outputs = self.bert(input_ids=text[:, 0], attention_mask=text[:, 1], token_type_ids=text[:, 2])

        text_seq = text_outputs["last_hidden_state"]
        text_rep = text_outputs["pooler_output"]

        if audio.shape[-1] != self.hidden_dim:
            audio = self.hidden_dim_alignment_audio(audio)
        if video.shape[-1] != self.hidden_dim:
            video = self.hidden_dim_alignment_video(video)

        video_seq = video
        audio_seq = audio

        video_mask = (
            torch.sum(
                video.ne(torch.zeros(video[0].shape[-1]).to(device)).int(),
                dim=-1,
            )
            / video[0].shape[-1]
        )
        video_mask_len = torch.sum(video_mask, dim=1, keepdim=True)
        video_mask_len = torch.where(
            video_mask_len > 0.5,
            video_mask_len,
            torch.ones([1]).to(device),
        )
        video_masked_output = torch.mul(video_mask.unsqueeze(2), video_seq)
        video_rep = torch.sum(video_masked_output, dim=1, keepdim=False) / video_mask_len

        audio_mask = (
            torch.sum(
                audio.ne(torch.zeros(audio[0].shape[-1]).to(device)).int(),
                dim=-1,
            )
            / audio[0].shape[-1]
        )
        audio_mask_len = torch.sum(audio_mask, dim=1, keepdim=True)
        audio_masked_output = torch.mul(audio_mask.unsqueeze(2), audio_seq)
        audio_rep = torch.sum(audio_masked_output, dim=1, keepdim=False) / audio_mask_len

        mult_text_seq, mult_audio_seq, mult_video_seq = self.mult_layer(
            text_feats=text_seq, video_feats=video_seq, audio_feats=audio_seq
        )

        mult_audio_seq = self.mult_audio(mult_audio_seq)
        mult_audio_seq = mult_audio_seq.permute(1, 2, 0)
        mult_audio_masked = torch.mul(audio_mask.unsqueeze(2), mult_audio_seq)
        mult_audio_rep = torch.sum(mult_audio_masked, dim=1, keepdim=False) / audio_mask_len

        mult_video_seq = self.mult_video(mult_video_seq)
        mult_video_seq = mult_video_seq.permute(1, 2, 0)
        mult_video_masked = torch.mul(video_mask.unsqueeze(2), mult_video_seq)
        mult_video_rep = torch.sum(mult_video_masked, dim=1, keepdim=False) / video_mask_len

        text_mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)
        mult_text_seq = self.mult_text(mult_text_seq)
        mult_text_seq = mult_text_seq.permute(1, 2, 0)
        mult_text_masked = torch.mul(bert_sent_mask.unsqueeze(2), mult_text_seq)
        mult_text_rep = torch.sum(mult_text_masked, dim=1, keepdim=False) / text_mask_len

        shallow_seq = torch.cat([mult_text_seq, mult_video_seq, mult_audio_seq], dim=1)

        all_reps = torch.stack((text_rep, video_rep, audio_rep, mult_text_rep, mult_audio_rep, mult_video_rep), dim=0)
        deep_rep = self.sdif_layer(all_reps, shallow_seq, bert_sent_mask, video_mask, audio_mask)
        logits = self.fusion(deep_rep)

        return logits
