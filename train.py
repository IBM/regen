""" Training and evaluation of ReGen models
"""
import logging.handlers
import logging
import configargparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AdamW

from dataset.dataloaders import get_dataloaders
from model.model import get_model, get_tokenizer
from utils.utils import mkdirs, log_versions, vars2str
from utils.state import get_state, get_model_from_state, get_optimizer_from_state, checkpoints
from utils.tensorboard import get_tensorboard
from utils.timing import Progress as Timer
from utils.sequence import shift_target_inputs_to_labels
from utils.evaluate import validate, generate
from utils.clean import cleanup
from loss.losses import SCST

import opts
import cfg


# ------------------------------
# Logger
# ------------------------------
# logger
logger = logging.getLogger()  # get logger
logger.setLevel(logging.INFO)
logger.propagate = False  # do not propagate logs to previously defined root loggers (if any) -- avoids some issues w/ 3rd party libs
formatter = logging.Formatter('%(asctime)s - %(levelname)s(%(name)s): %(message)s')
# console
consH = logging.StreamHandler()
consH.setFormatter(formatter)
consH.setLevel(logging.INFO)
logger.addHandler(consH)
# file handler
request_file_handler = True
log = logger


def main(args):

    # console handler
    formatter_rank = logging.Formatter(f'[{args.local_rank}] %(asctime)s - %(levelname)s(%(name)s): %(message)s')
    consH.setFormatter(formatter_rank)

    # file handler
    if request_file_handler:

        logdir = args.log_dir
        mkdirs([logdir], logger=log)
        log_filename = os.path.join(logdir, f'train.{args.jid:02d}.rank.{args.local_rank}.log')
        file_handler = logging.FileHandler(log_filename, mode='w')
        file_handler.setFormatter(formatter_rank)
        file_handler.setLevel(logging.INFO)
        # add handler
        logger.addHandler(file_handler)
        log.info('# File logger started')

    # derived vars
    rank_zero = True if args.local_rank == 0 else False
    log.info(f'# world_size: {args.world_size} local_rank: {args.local_rank} rank_zero: {rank_zero} args.jid: {args.jid}')

    # Distributed
    dist.init_process_group("nccl")
    log.info("# Distributed setup completed")

    log_versions(log, ['python', 'torch', 'numpy', 'transformers'])

    # pretty print args
    log.info(f'# args: {vars2str(args)}')

    # dirs
    if rank_zero:
        mkdirs([args.output_dir,
                args.tensorboard_dir,
                args.json_logger_dir,
                args.checkpoint_dir,
                args.checkpoint_eval_dir],
               logger=log)

    # Device (gpu/cpu)
    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')
    log.info(f'# Device: {device}')

    # properly load saved models for each worker
    map_location = {'cuda:0': f'{device}'}

    # Load state
    state, state_path = get_state(args)

    # global iteration to start from
    iteration = state.get('iteration', 0)  # 0-based
    epoch_prev = state.get('epoch', 0.0)
    iteration_prev = iteration
    log.info(f"# state: iteration: {iteration}  epoch:{epoch_prev}")

    # cleanup
    if rank_zero:
        skip = [args.scst_checkpoint_id] if args.scst else None
        cleanup(args, remove_files=True, remove_all_past=False, skip=skip)

    # tensorboard
    if rank_zero:
        results_json_log = os.path.join(args.json_logger_dir, f'results.json.{args.jid:02d}.rank.{args.local_rank}.log')
        args.results_json_log = results_json_log
        writer_tb = get_tensorboard(args, iteration, logger)

    # tokenizer
    tokenizer, _ = get_tokenizer(args.model)

    # dataloaders
    dataloaders = get_dataloaders(args, tokenizer)
    train_dataloader = dataloaders['train']
    valid_dataloader = dataloaders['val']

    # model
    model_pre, _ = get_model(args.model)
    model_pre.resize_token_embeddings(len(train_dataloader.dataset.tokenizer))  # update for new tokens

    model = get_model_from_state(state, model_pre, map_location)
    model.to(device)

    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    log.info("# Model loading from state")

    # log.info(f'# model: {model}')

    # From Training and fine-tuning in HF doc: https://huggingface.co/transformers/training.html#pytorch
    '''
        The optimizer allows us to apply different hyperpameters for specific parameter groups. For example,
        we can apply weight decay to all parameters other than bias and layer normalization terms:

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
    '''
    if not args.scst:
        optimizer = AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr)  # enables to try different optimizers
        # only start from an optimizer from a previous SCST job
        if args.scst_jid > 1:
            optimizer = get_optimizer_from_state(state, optimizer, map_location)

    # data iterator
    train_iterator, epochN, epochF = iter(train_dataloader), len(train_dataloader), 0.0  # note: 'epochF' amount of epoch processed (as a float)

    # adjust number of minib to process given option
    if args.job_num_minib <= -1:
        # number of epochs to run
        factor = -args.job_num_minib
        args.job_num_minib = epochN * factor
        log.info(f'# ADJUST: Number of iteration per job:  set to {factor} epoch(s) = {args.job_num_minib} minib [{factor}*{epochN}]')

    # adjust number of minib for checkpointing
    if args.checkpoint_every <= -1:
        # number of minib for checkpointing
        factor = -args.checkpoint_every
        args.checkpoint_every = epochN * factor
        log.info(f'# ADJUST: Number of iterations for checkpointing:  set to {factor} epoch(s) = {args.checkpoint_every} minib [{factor}*{epochN}]')

    if args.scst:
        log.info('# metric   : weight -- SCST')
        for metric, weight in args.use_metric_weight:
            log.info(f'# {metric:>8s} : {weight}')

    # Main loop
    itr_loss, itr = 0.0, -1
    log.info("# Starting Training Loop")

    # validation on start
    if rank_zero and args.valid_on_start:
        log.info("# Validation on start")
        valid_loss = validate(valid_dataloader, model, device, args.src, msg=f'itr:{iteration}')
        log.info(f'# START: validation loss: {valid_loss}')

    # training progress timer
    timer = Timer(epochN, everyN=args.timer_progress_every, logger=log, name='train_timer', iteration=iteration)

    while True:

        try:
            batch = next(train_iterator)

        except StopIteration:

            # reset training iterator and timers
            avg_loss = itr_loss / float(itr + 1)  # itr is 0-base
            log.info(f'# Train :: Epoch:{iteration / float(epochN):.3f} iteration:{iteration}  avg_loss:{avg_loss:.7f}')

            train_iterator, epochN = iter(train_dataloader), len(train_dataloader)
            batch = next(train_iterator)

            itr_loss, itr = 0.0, -1
            log.info("# starting another training epoch")
            timer = Timer(epochN, everyN=args.timer_progress_every, logger=log, name='train_timer', iteration=iteration)

        itr += 1  # 0 at begining of 'epoch'
        iteration += 1  # 1 at begining

        epochF = (iteration - iteration_prev) / epochN + epoch_prev  # epoch in floating point number (independent of batch_size and GPU number)
        # log.info(f'# iteration: {iteration} itr {itr}')

        model.train()

        # get source, targets (and respective masks), and labels
        source_ids, source_mask, target_ids, target_mask, _ = (x.to(device) for x in batch)
        labels = shift_target_inputs_to_labels(target_ids, tokenizer.pad_token_id, device)

        # fw
        if not args.scst:  # ce

            output = model(
                input_ids=source_ids,
                attention_mask=source_mask,
                decoder_input_ids=target_ids,
                decoder_attention_mask=target_mask,
                labels=labels
            )

        else:  # scst

            output = SCST(
                model,
                source_ids,
                source_mask,
                target_ids,
                target_mask,
                labels,
                tokenizer,
                iteration,
                args)

        loss = output.loss  # loss is averaged by minib
        loss_float = loss.item()
        itr_loss += loss_float

        # bw
        norm_loss = loss / args.train_minib_aggregate
        norm_loss.backward()

        # grad clipping?
        if args.grad_clip:
            _model = model.module if isinstance(model, DDP) else model
            total_grad_norm = torch.nn.utils.clip_grad_norm_(_model.parameters(),
                                                             args.clip_max_norm,
                                                             args.clip_norm_type)
            if rank_zero and iteration % args.log_every == 0:
                log.info(f'# total_grad_norm {total_grad_norm} [max_norm={args.clip_max_norm}, norm_type={args.clip_norm_type}]')

        if (itr + 1) % args.train_minib_aggregate == 0:

            optimizer.step()
            optimizer.zero_grad()

        # update progress logger
        timer.progress(iteration, header=f'# it:{iteration} ')

        if rank_zero and iteration % args.log_every == 0:
            avg_loss = itr_loss / float(itr + 1)  # itr is 0-base
            log.info(f"# Train :: it:{iteration} [epoch:{iteration / float(epochN):.3f}] loss: {loss.item():.7f}  avg_loss:{avg_loss}")
            writer_tb.add_scalar('train/loss', loss_float, iteration)

        if rank_zero and iteration % args.checkpoint_every == 0:
            checkpoints(args, epochF, iteration, model, optimizer, checkpoint_type='offline_eval')

        if args.job_num_minib > 0 and iteration % args.job_num_minib == 0:
            break

    # End
    if rank_zero and args.valid_on_end:
        valid_loss = validate(valid_dataloader, model, device, args.src, msg=f'itr:{iteration}')
        log.info(f'# END: validation loss: {valid_loss}')
        writer_tb.add_scalar('validation/loss', valid_loss, iteration)

    if rank_zero and args.valid_generate_on_end:
        bleu_score, hyps, refs = generate(valid_dataloader, model, device, args.src, args.valid_beam_size, args.valid_max_length, msg=f'itr:{iteration}')
        log.info(f'# generation bleu score : {bleu_score}')

    if rank_zero:
        # save checkpoint on end
        checkpoints(args, epochF, iteration, model, optimizer, last=True)

    torch.distributed.barrier()

    log.info("# done")


if __name__ == "__main__":

    # Parsing arguments
    p = configargparse.ArgParser(description='Training Arguments')

    # require config file
    p.add("-c", "--config", required=True, is_config_file=True, help="config file path")

    # add arguments
    p = opts.add_arguments(p)

    # add more arguments:
    # distributed args
    p.add("--local_rank", type=int, default=-1, help="rank process ID")
    p.add("--world_size", type=int, default=-1, help="number of processes")

    args = p.parse_args()

    log.info(p.format_values())

    # add some variables:
    args.dataset_choices = cfg.dataset_choices
    args.distributed = True  # some code may check if distributed
    if args.scst:
        if args.jid <= args.scst_checkpoint_id:
            raise ValueError(f"jobid args.jid={args.jid} for SCST must be greater than jobid of starting model args.scst_checkpoint_id={args.scst_checkpoint_id}")

        args.scst_jid = args.jid - args.scst_checkpoint_id  # "job_id" for scst

    # check arguments, add some more too
    args = opts.check_and_add_arguments(args)

    main(args)
