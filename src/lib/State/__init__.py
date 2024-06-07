from typing import Union, List, Tuple
import torch


class State:
    def __init__(
        self,
        encoder_output: Union[torch.Tensor, List, Tuple] = None,
        encoder_mask: Union[torch.Tensor, List, Tuple] = None,
        **kwargs,
    ):
        self.encoder_output = encoder_output
        self.encoder_mask = encoder_mask
        self._decode_length = 0

    @property
    def num_samples(self):
        if self.encoder_output is not None:
            return self.encoder_output.size(0)
        else:
            return None

    @property
    def decode_length(self):
        return self._decode_length

    @decode_length.setter
    def decode_length(self, value):
        self._decode_length = value

    def _reorder_state(
        self,
        state: Union[torch.Tensor, list, tuple],
        indices: torch.LongTensor,
        dim: int = 0,
    ):
        if isinstance(state, torch.Tensor):
            state = state.index_select(index=indices, dim=dim)
        elif isinstance(state, list):
            for i in range(len(state)):
                assert state[i] is not None
                state[i] = self._reorder_state(state[i], indices, dim)
        elif isinstance(state, tuple):
            tmp_list = []
            for i in range(len(state)):
                assert state[i] is not None
                tmp_list.append(self._reorder_state(state[i], indices, dim))
            state = tuple(tmp_list)
        else:
            raise TypeError(f"Cannot reorder data of type:{type(state)}")

        return state

    def reorder_state(self, indices: torch.LongTensor):
        if self.encoder_mask is not None:
            self.encoder_mask = self._reorder_state(self.encoder_mask, indices)
        if self.encoder_output is not None:
            self.encoder_output = self._reorder_state(self.encoder_output, indices)
