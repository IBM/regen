import logging
import torch

logger = logging.getLogger(__name__)
log = logger


def shift_target_inputs_to_labels(tgt_input_ids, pad_token_id, device):
    """
    <bos> word1 word2 word3 <eos> (target input)
    word1 word2 word3 <eos> <pad> (target label)
    from https://github.com/znculee/finetune-transformers (MIT License)
    """
    batch_pads = torch.empty(
        tgt_input_ids.shape[0], 1,
        dtype=tgt_input_ids.dtype,
        device=device
    ).fill_(pad_token_id)
    labels = torch.cat((tgt_input_ids[:, 1:], batch_pads), dim=1)

    # masking pad tokens in labels with -100 for CE to ignore
    pad_mask = labels == pad_token_id
    labels[pad_mask] = -100

    return labels
