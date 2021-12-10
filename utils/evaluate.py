""" Evaluation
"""
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import sacrebleu

from utils.timing import Progress as Timer
from utils.sequence import shift_target_inputs_to_labels

import logging

logger = logging.getLogger(__name__)
log = logger


def validate(val_dl, model, device, src, msg=''):
    ''' Evaluates model against validation set.'''

    valid_dataset = val_dl.dataset

    # validation
    with torch.no_grad():
        model.eval()

        valN = len(val_dl)
        valid_sum_loss = 0.0

        # Timer
        timer_eval = Timer(valN, percent=5, zero_based=True, logger=log, name='validate_timer')

        for itr, batch in enumerate(val_dl):
            source_ids, source_mask, target_ids, target_mask, _ = (x.to(device) for x in batch)

            labels = shift_target_inputs_to_labels(target_ids, valid_dataset.tokenizer.pad_token_id, device)

            # need to work on inputs
            output = model(
                input_ids=source_ids,
                attention_mask=source_mask,
                decoder_input_ids=target_ids,
                decoder_attention_mask=target_mask,
                labels=labels
            )

            valid_loss = output[0]
            valid_sum_loss += valid_loss * source_ids.shape[0]

            # update progress logger
            timer_eval.progress(itr, header=f'# VALIDATE: {msg}')

    # model in training mode
    model.train()

    valid_measure = valid_sum_loss.item() / len(valid_dataset)
    logger.info(f'# validation loss: {valid_measure}')

    return valid_measure


def generate(eval_dl, model, device, src, beam_size, max_length, msg='', use_targets=True):
    ''' Generates output sequences starting from validation set inputs

    Arguments:
      eval_dl: evaluation dataloader
      model: model to evaluate
      device: device ('cpu' or 'cuda', ...)
      src: modality source 'A' or 'B'
      beam_size: beam_size for model.gnenerate()
      max length: max lenght for model.generate()
      msg: message to output
      use_targets: use targets to compute some evaluation score/metric

    Outputs:
     hypotheses, targets, valid_measure, meta info
    where
     hypotheses: List of hypotheses
     targets: List of targets
     valid_measure: scalar for validation measure
     categories: List of categories
    '''

    eval_dataset = eval_dl.dataset

    hypotheses = []
    targets = []
    meta_infos = []

    gen_model = model.module if isinstance(model, DDP) else model

    # validation
    with torch.no_grad():

        model.eval()

        evalN = len(eval_dl)

        # Timer
        # timer_eval = Timer(evalN, percent=10, zero_based=False, logger=log, name='generate_timer')
        timer_eval = Timer(evalN, everyN=10, zero_based=True, logger=log, name='generate_timer')

        for itr, batch in enumerate(eval_dl):

            source_ids, source_mask, target_ids, target_mask, sample_idx_batch = (x.to(device) for x in batch)

            output_ids = gen_model.generate(
                source_ids,
                attention_mask=source_mask,
                num_beams=beam_size,
                max_length=max_length,
                # early_stopping=True
            )

            # prepare hypotheses
            for seq_ids in output_ids.to('cpu').numpy().tolist():
                seq_toks = eval_dataset.tokenizer.decode(
                    seq_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                hypotheses.append(seq_toks)

            # prepare targets
            for seq_ids in target_ids.to('cpu').numpy().tolist():
                seq_toks = eval_dataset.tokenizer.decode(
                    seq_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                targets.append(seq_toks)

            # get meta infos
            meta_infos += [eval_dataset.get_meta(idx) for idx in sample_idx_batch.tolist()]

            # update progress logger
            timer_eval.progress(itr, header=f'# GENERATE: {msg} ')

        if use_targets:
            bleu = sacrebleu.corpus_bleu(hypotheses, [targets], force=True)
            valid_measure = -bleu.score
            logger.info(f'# validation BLEU: {bleu.score} [on targets]')
        else:
            valid_measure = 0

    # model in training mode
    model.train()

    return hypotheses, targets, valid_measure, meta_infos


def get_tokens_seq_from_token_ids(token_ids, tokenizer):
    ''' Retruns tokens from a sequence of token ids '''

    tokens = []

    for seq_ids in token_ids.to('cpu').numpy().tolist():
        seq_toks = tokenizer.decode(
            seq_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        tokens.append(seq_toks)

    return tokens
