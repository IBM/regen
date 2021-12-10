""" Evaluation of models for validation and generation
"""
import logging.handlers
import logging
import configargparse
import os

from addict import Dict as adict
import torch

from dataset.dataloaders import get_dataloader
from model.model import get_model, get_tokenizer
from utils.utils import mkdirs, log_versions, vars2str
from utils.state import load_state, get_model_from_state, get_args_from_state
from utils.tensorboard import get_tensorboard
from utils.evaluate import validate, generate
from utils.utils import save_json
from utils.utils_webnlg import (save_webnlg_text, save_webnlg_rdf)
from utils.utils_tekgen import save_tekgen_rdf

# ------------------------------
# Logger
# ------------------------------
# logger
logger = logging.getLogger()  # get (root) logger
logger.setLevel(logging.INFO)
logger.propagate = False  # do not propagate logs to previously defined root logger (if any).
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

    # file handler
    if request_file_handler:

        logdir = args.log_dir
        mkdirs([logdir], logger=log)
        log_filename = os.path.join(logdir, 'generate.log')
        file_handler = logging.FileHandler(log_filename, mode='w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        # add handler
        logger.addHandler(file_handler)
        log.info('# File logger started')

    log_versions(log, ['torch', 'transformers', 'numpy'])

    # pretty print args
    log.info(f'# args: {vars2str(args)}')

    # dirs
    mkdirs([args.output_dir,
            args.tensorboard_dir,
            args.json_logger_dir,
            args.generate_dir],
           logger=log)

    # Device (gpu/cpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'# Device: {device}')

    # properly load saved models to proper device
    map_location = {'cuda:0': f'{device}'}

    # Load state
    state = None

    if args.state_path:
        state = load_state(args.state_path)
    else:
        raise RuntimeError('no way to load model for evaluation')

    args_state = get_args_from_state(state)

    # Transfer argument from train args
    args.model = args_state.model
    args.src = args_state.src
    if args.prepare != args_state.prepare:
        log.warning(f"prepare from CLI and state do not match: [CLI] '{args.prepare}' != '{args_state.prepare}' [state] -> for evaluation we use '{args.prepare}' [CLI]")

    prepare_allowed = ['webnlg', 'tekgen-official']
    if args.prepare not in prepare_allowed:
        raise ValueError(f"For now we only allow args.prepare in '{prepare_allowed}', not '{args.prepare}'")

    args.prepare_permutation = "none"  # no permutations in testing

    # global iteration to start from
    iteration = state.get('iteration', 0)  # 0-based
    epochF = state.get('epoch', 0.0)  # 0-based

    log.info(f'# iteration: {iteration} epoch: {epochF}')

    # tensorboard
    results_json_log = os.path.join(args.json_logger_dir, 'results.json.log')
    args.results_json_log = results_json_log
    writer_tb = get_tensorboard(args, iteration, logger)

    # tokenizer
    tokenizer, _ = get_tokenizer(args.model)

    # Make sure we have the correct task and src
    if args.split == 'testA':  # A->B  t2g
        if args.src == 'B':  # graph
            raise RuntimeError(f"model trained with 'B' source (arg.src={args.src}) cannot be used on task '{args.split}'")
        elif args.src == 'A+B':  # text + graph
            log.info(f"# OVERRIDE: args.src='{args.src}' -> changed to 'A' to work with test task '{args.split}'")
            args.src = 'A'

    elif args.split == 'testB':  # B->A  g2t
        if args.src == 'A':  # text
            raise RuntimeError(f"model trained with 'A' source (arg.src={args.src}) cannot be used on task '{args.split}'")
        elif args.src == 'A+B':  # text + graph
            log.info(f"# OVERRIDE: args.src='{args.src}' -> changed to 'B' to work with test task '{args.split}'")
            args.src = 'B'

    elif args.split == 'valA':  # A->B  t2g
        if args.src == 'B':  # graph
            raise RuntimeError(f"model trained with 'B' source (arg.src={args.src}) cannot be used on task '{args.split}'")
        elif args.src == 'A+B':  # text + graph
            log.info(f"# OVERRIDE: args.src='{args.src}' -> changed to 'A' to work with test task '{args.split}'")
            args.src = 'A'

    elif args.split == 'valB':  # B->A  g2t
        if args.src == 'A':  # text
            raise RuntimeError(f"model trained with 'A' source (arg.src={args.src}) cannot be used on task '{args.split}'")
        elif args.src == 'A+B':  # text + graph
            log.info(f"# OVERRIDE: args.src='{args.src}' -> changed to 'B' to work with test task '{args.split}'")
            args.src = 'B'

    else:
        raise RuntimeError(f"unknown split '{args.split}' should be 'testA', 'testB', 'valA', or 'valB'")

    # dataloaders
    # @debug dataloader = get_dataloader(args, tokenizer, args.split, list(range(10)))
    split = 'val' if (args.split == 'valA' or args.split == 'valB') else args.split

    dataloader = get_dataloader(args, tokenizer, split)
    log.info(f"# for dataset '{args.dataset}': restricting samples w/ args.subset_fraction={args.subset_fraction}")

    # model
    model, _ = get_model(args.model)
    model.resize_token_embeddings(len(dataloader.dataset.tokenizer))  # update for new tokens (trained and will be loaded below)
    model.to(device)

    # model from state
    log.info("# Model loading from state")
    model = get_model_from_state(state, model, map_location)
    # log.info(f'# model: {model}')

    # validation
    if args.validate:
        valid_loss = validate(dataloader, model, device, args.src, msg=f'itr:{iteration}')
        log.info(f'# validation loss: {valid_loss}')
        writer_tb.add_scalar('valid/loss', valid_loss, iteration)

    # what type of source from args.split
    is_srcA = (args.split in ['testA', 'valA'])
    is_srcB = (args.split in ['testB', 'valB'])

    # generation
    if args.generate:
        hyps, targets, bleu_score, metas = generate(dataloader, model, device, args.src, args.valid_beam_size,
                                                    args.valid_max_length, msg=f'itr:{iteration}')

        log.info(f'# generation bleu score : {bleu_score} [validation between hypotheses and targets]')

        cfg_dict = adict()
        cfg_dict.dataset = args.dataset
        cfg_dict.prepare = args.prepare
        cfg_dict.model = args.model
        cfg_dict.split = args.split
        cfg_dict.src = args.src
        cfg_dict.valid_beam_size = args.valid_beam_size
        cfg_dict.valid_max_length = args.valid_max_length

        hyps_cfg = os.path.join(args.generate_dir, 'hyps.cfg.json')

        hyps_json = os.path.join(args.generate_dir, 'hyps.json')
        refs_json = os.path.join(args.generate_dir, 'targets.json')
        meta_json = os.path.join(args.generate_dir, 'metas.json')

        save_json(hyps_cfg, cfg_dict, msg='config: ', logger=log)

        save_json(hyps_json, hyps, msg='hypotheses: ', logger=log)
        save_json(refs_json, targets, msg='targets: ', logger=log)
        save_json(meta_json, metas, msg='meta info: ', logger=log)

        if args.src == 'A' and is_srcA:  # A->B

            # Note: For WebNLG, targets and references for evaluation are *not* the same thing.
            # - targets are coming from the evaluation dataloader (after pre-processing)
            # - references are coming directly from the WebNLG Challenge .xml reference files and are *the* ground truth (un-processed)
            if args.dataset == 'webnlg':
                _ = save_webnlg_rdf(hyps, targets, metas, args.generate_dir, args.prepare)
            elif args.dataset == 'tekgen':
                _ = save_tekgen_rdf(hyps, targets, metas, args.generate_dir, args.prepare)
            else:
                raise ValueError(f"dataset '{args.dataset}' is not supported")

        elif args.src == 'B' and is_srcB:  # B->A

            _ = save_webnlg_text(hyps, targets, args.generate_dir)

        else:
            log.info(f'something is wrong args.src={args.src} and args.split={args.split} do not match (src for dataset does not match split requested)')

    log.info("# done")


if __name__ == "__main__":

    import opts

    # Note:
    #
    # WebNLG, has 3 possible evaluation sets: 'val', 'testA', 'testB'
    # - 'val' can accomodate both A->B (t2g) and B->A (g2t) evaluations
    # - 'testA' cam accomodate 'A->B' (t2g) only
    # - 'testB' can accomodate 'B->A' (g2t) only
    #
    # Models may have been trained given 3 possible scenarios: src='A','B', or 'A+B' (w/ A=text, B=graph)
    #
    # We have therefore multiple cases:
    # - 'testA': model must have src='A' to make sense (model trained for 'A->B' (t2g) to do a 'A->B' (t2g) evaluation)
    # - 'testB': model must have src='B' to make sense (model trained for 'B->A' (g2t) to do a 'B->A' (g2t) evaluation)
    # - 'val', model can have src='A', 'B', or 'A+B', we need to narrow to src='A' or 'B' for evaluation
    #    Therefore we are allowing for requested src:
    #    - if model src='A' or 'A+B' and request src='A' -> ok -- we will use the 'valA' moniker for this evaluation
    #    - if model src='B' or 'A+B' and request src='B' -> ok -- we will use the 'valB' moniker for this evaluation
    #

    # Parsing arguments
    p = configargparse.ArgParser(description='Generate Arguments')

    # require config file?
    p.add("-c", "--config", required=False, is_config_file=True, help="config file path")

    p = opts.add_arguments_generate(p)
    args = p.parse_args()

    # checks
    args = opts.check_and_add_arguments_generate(args)

    main(args)
