import torch
from torch._six import inf
from typing import Union, Iterable

import logging

logger = logging.getLogger(__name__)
log = logger

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def grad_norm_(parameters: _tensor_or_tensors, norm_type: float = 2.0) -> torch.Tensor:
    '''
    Computes gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).

    Note: code based on  https://pytorch.org/docs/1.7.1/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    '''
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)

    if len(parameters) == 0:
        return torch.tensor(0.)

    device = parameters[0].grad.device

    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)

    return total_norm
