import torch
import torch.nn as nn
from typing import Union, Tuple

from src.lib.State import State


class Seq2SeqDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tokens: "torch.LongTensor", state: State, **kwargs):
        raise NotImplemented

    def reorder_states(self, indices: torch.LongTensor, states):
        assert isinstance(states, State), f"`states` should be of type State instead of {type(states)}"
        states.reorder_state(indices)

    def init_state(
        self,
        encoder_output: Union[torch.Tensor, list, tuple],
        encoder_mask: Union[torch.Tensor, list, tuple],
    ):
        state = State(encoder_output, encoder_mask)
        return state

    def decode(self, tokens: torch.LongTensor, state: State) -> torch.FloatTensor:
        outputs = self(state=state, tokens=tokens)
        if isinstance(outputs, torch.Tensor):
            return outputs[:, -1]
        else:
            raise RuntimeError(
                "Unrecognized output from the `forward()` function. Please override the `decode()` function."
            )
