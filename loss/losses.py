import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from addict import Dict as adict

from utils.evaluate import get_tokens_seq_from_token_ids
from utils.rewards import self_critical_reward_text, self_critical_reward_triple, self_critical_reward_AB
from utils.model_sampling import T5Sampling
from utils.scst import log_scst

import logging

logger = logging.getLogger(__name__)
log = logger


def SCST(model, source_ids, source_mask, target_ids, target_mask, labels, tokenizer, iteration, args, verbose=False):
    '''
    SCST forward pass
    '''
    gen_model = model.module if isinstance(model, DDP) else model

    max_length = target_ids.size(1)  # target_ids = [batch_size, seq_len]

    gen_model.eval()
    with torch.no_grad():
        # greedy max decode
        greedy_ids = gen_model.generate(
            source_ids,
            attention_mask=source_mask,
            max_length=max_length
        )
    gen_model.train()

    # prepare hypotheses
    greedy_seq = get_tokens_seq_from_token_ids(greedy_ids, tokenizer)
    target_seq = get_tokens_seq_from_token_ids(target_ids, tokenizer)  # "ground truth"

    # sampling
    model_sampling = T5Sampling(args, gen_model)

    do_sample = True
    output = model_sampling(
        input_ids=source_ids,
        attention_mask=source_mask,
        decoder_input_ids=target_ids,
        decoder_attention_mask=target_mask,
        labels=labels,
        return_dict=True,
        do_sample=do_sample
    )
    # output: custom SamplingSeq2SeqLMOutput

    # We need the sequence of tokens... out of the fw sampling:
    # output.sequences : [batch_size, max_length]

    with torch.no_grad():

        # remove the first token (decoder_start_token_id) from the sampling_outputs.sequences
        sample_seq = tokenizer.batch_decode(output.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        if args.src == 'A':  # t2g text2rdf
            rewards = self_critical_reward_triple(greedy_seq, target_seq, sample_seq, args)

        elif args.src == 'B':  # g2t rdf2text
            rewards = self_critical_reward_text(greedy_seq, target_seq, sample_seq, args)

        elif args.src == 'A+B':  # [t2g + g2t] [text2rdf + rdf2txt]

            rewards = self_critical_reward_AB(greedy_seq, target_seq, sample_seq, args)

        else:
            raise ValueError("# invalid source '{source}'")

        if verbose:
            log_scst(greedy_seq, target_seq, sample_seq, rewards, msg=f'itr:{iteration}')

    # scst loss = - logprobs * reward * mask
    #      loss = sum(loss) / sum(mask)    where sum(mask) means sum of entries

    # need to have logprobs
    logprobs = output.sample_logprobs  # [bs x max_seq_len x 1]

    reward = rewards['reward']
    reward = torch.from_numpy(reward).to(logprobs.device).float()  # [bs]

    reward = torch.unsqueeze(reward, 1)  # [bs, 1]
    reward = torch.unsqueeze(reward, 2)  # [bs, 1, 1]

    pad_token_id = gen_model.config.pad_token_id
    mask = (output.sequences != pad_token_id).float()  # [bs, max_length]
    mask = torch.unsqueeze(mask, 2)  # [bs, max_length, 1]

    # REINFORCE J(theta) for gradient descent
    scst_loss = -logprobs * mask * reward
    loss = torch.sum(scst_loss) / torch.sum(mask)

    output = adict()
    output.loss = loss
    output.reward = reward

    return output
