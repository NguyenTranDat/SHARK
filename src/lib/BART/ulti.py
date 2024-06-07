import torch


def get_utt_representation(all_word_state, utt_prefix_ids, dia_utt_num):
    bz, _, h = all_word_state.size()  # bsz x max_word_len x hidden_size

    output = all_word_state.gather(
        index=utt_prefix_ids.unsqueeze(2).repeat(1, 1, all_word_state.size(-1)), dim=1
    )  # bsz x max_utt_len x hidden_size
    # batch_size = dia_utt_num.shape[0]
    batch_size = dia_utt_num.shape[0]
    broad_cast_seq_len = torch.arange(output.size(1)).expand(batch_size, -1).to(dia_utt_num)
    mask = broad_cast_seq_len < dia_utt_num.unsqueeze(1)
    utt_mask = mask.eq(0)  # bsz x max_utt_len
    utt_mask_ = utt_mask.unsqueeze(2).repeat(1, 1, output.size(-1))
    output = output.masked_fill(utt_mask_, 0)

    # cls_tokens, _ = torch.max(hidden_states, dim=1)  # max pooling

    return output  # bsz x max_utt_len(35) x hidden_size
