import torch
import inspect
from collections import Counter, namedtuple
from typing import Optional


CheckRes = namedtuple("CheckRes", ["missing", "unused", "duplicated", "required", "all_needed", "varargs"])


def seq_len_to_mask(seq_len, max_len: Optional[int] = None):
    batch_size = seq_len.shape[0]
    broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
    mask = broad_cast_seq_len < seq_len.unsqueeze(1)
    return mask


def get_utt_representation(all_word_state, utt_prefix_ids, dia_utt_num):
    bz, _, h = all_word_state.size()  # bsz x max_word_len x hidden_size

    output = all_word_state.gather(
        index=utt_prefix_ids.unsqueeze(2).repeat(1, 1, all_word_state.size(-1)), dim=1
    )  # bsz x max_utt_len x hidden_size
    # batch_size = dia_utt_num.shape[0]
    utt_mask = seq_len_to_mask(dia_utt_num, max_len=output.shape[1]).eq(0)  # bsz x max_utt_len
    utt_mask_ = utt_mask.unsqueeze(2).repeat(1, 1, output.size(-1))
    output = output.masked_fill(utt_mask_, 0)

    # cls_tokens, _ = torch.max(hidden_states, dim=1)  # max pooling

    return output  # bsz x max_utt_len(35) x hidden_size


def get_func_signature(func):
    """

    Given a function or method, return its signature.
    For example:
    (1) function
        def func(a, b='a', *args):
            xxxx
        get_func_signature(func) # 'func(a, b='a', *args)'
    (2) method
        class Demo:
            def __init__(self):
                xxx
            def forward(self, a, b='a', **args)
        demo = Demo()
        get_func_signature(demo.forward) # 'Demo.forward(self, a, b='a', **args)'
    :param func: a function or a method
    :return: str or None
    """
    if inspect.ismethod(func):
        class_name = func.__self__.__class__.__name__
        signature = inspect.signature(func)
        signature_str = str(signature)
        if len(signature_str) > 2:
            _self = "(self, "
        else:
            _self = "(self"
        signature_str = class_name + "." + func.__name__ + _self + signature_str[1:]
        return signature_str
    elif inspect.isfunction(func):
        signature = inspect.signature(func)
        signature_str = str(signature)
        signature_str = func.__name__ + signature_str
        return signature_str


def _check_arg_dict_list(func, args):
    if isinstance(args, dict):
        arg_dict_list = [args]
    else:
        arg_dict_list = args
    assert callable(func) and isinstance(arg_dict_list, (list, tuple))
    assert len(arg_dict_list) > 0 and isinstance(arg_dict_list[0], dict)
    spect = inspect.getfullargspec(func)
    all_args = set([arg for arg in spect.args if arg != "self"])
    defaults = []
    if spect.defaults is not None:
        defaults = [arg for arg in spect.defaults]
    start_idx = len(spect.args) - len(defaults)
    default_args = set(spect.args[start_idx:])
    require_args = all_args - default_args
    input_arg_count = Counter()
    for arg_dict in arg_dict_list:
        input_arg_count.update(arg_dict.keys())
    duplicated = [name for name, val in input_arg_count.items() if val > 1]
    input_args = set(input_arg_count.keys())
    missing = list(require_args - input_args)
    unused = list(input_args - all_args)
    varargs = [] if not spect.varargs else [spect.varargs]
    return CheckRes(
        missing=missing,
        unused=unused,
        duplicated=duplicated,
        required=list(require_args),
        all_needed=list(all_args),
        varargs=varargs,
    )


def _build_args(func, **kwargs):
    spect = inspect.getfullargspec(func)
    if spect.varkw is not None:
        return kwargs
    needed_args = set(spect.args)
    defaults = []
    if spect.defaults is not None:
        defaults = [arg for arg in spect.defaults]
    start_idx = len(spect.args) - len(defaults)
    output = {name: default for name, default in zip(spect.args[start_idx:], defaults)}
    output.update({name: val for name, val in kwargs.items() if name in needed_args})
    return output
