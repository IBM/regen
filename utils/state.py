import logging
import os
import random

import numpy as np
import torch

from utils.utils import read_pickle, save_pickle, vars2str

logger = logging.getLogger(__name__)
log = logger


def load_state(state_path, rebase_paths=False, silent=False):
    ''' Loads state from path, rebase internal paths to location of state_path
        rebasing paths:
          if pointing to a state_path from another expt, all the relative paths in the state will be broken.
          rebasing the paths is a way to fix that by replacing them with absolute paths based on state_path.
          For example:
            state_path = '../../expt/A00/output/checkpoints/state.20.pkl
            Assume state's absolute path is  /fs/expt/A00/output/checkpoints/state.20.pkl
            Then, all paths in state['paths'] will be changed to use /fs/expt/A00/output/checkpoints/{filename}
            as dirname, keeping the filename as is.
    '''

    state = None

    if os.path.exists(state_path):
        if not silent:
            log.info(f'# Loading state from {state_path}')
        state = read_pickle(state_path, logger=log)
        if not silent:
            log.info(f"# STATE: iteration: {state['iteration']}")
            log.info(f"# STATE: paths: {vars2str(state['paths'])}")
        if rebase_paths:
            if not silent:
                log.info("# STATE: rebasing paths requested")
            state_realpath = os.path.realpath(state_path)
            dir_path = os.path.dirname(state_realpath)
            for k, path in state['paths'].items():
                base = os.path.basename(path)
                state['paths'][k] = os.path.join(dir_path, base)
                log.info(f"# STATE: rebasing state['paths']['{k}']: {path} -> {state['paths'][k]}")
    else:
        raise RuntimeError(f'no state file {state_path}')

    return state


def get_state(args):
    ''' Gets state for training.
       Do we start from a previous checkpoint?
       if jid = 1 : no, returns empty dict and None for state path
       if jid > 1 : yes, from state.${jid-1}.pkl returns state dict and state path
    '''
    state = {}

    if args.jid > 1:  # start from previous

        prev_id = args.jid - 1
        state_path = os.path.join(args.checkpoint_dir, f'state.{prev_id}.pkl')

        if os.path.exists(state_path):

            log.info('# STATE: from previous checkpoint')
            state = read_pickle(state_path, logger=log)
            log.info(f"# STATE: prev iteration: {state['iteration']}")
            log.info(f"# STATE: prev paths: {vars2str(state['paths'])}")

            # Load RNGs from checkpoint
            log.info("# STATE: random seeds: loaded from previous state")
            np.random.set_state(state['seeds']['numpy_state'])
            random.setstate(state['seeds']['python_state'])
            torch.set_rng_state(state['seeds']['torch_state'])

            if torch.cuda.is_available():
                torch.cuda.set_rng_state(state['seeds']['cuda_state'])

        else:
            raise RuntimeError(f'no state file {state_path}')

    elif args.jid == 1:  # start from scratch

        log.info("# STATE: from scratch")

        # Random seeds
        seed = args.seed
        np_seed = seed
        torch_seed = seed
        random.seed(seed)  # python random
        np.random.seed(np_seed)  # numpy seed
        torch.manual_seed(torch_seed)  # torch seed

        log.info(f'# START: random seeds: random {seed}')
        log.info(f'# START: random seeds: numpy {np_seed}')
        log.info(f'# START: random seeds: torch {torch_seed}')

        if torch.cuda.is_available():
            # cuda random seed
            cuda_seed = seed
            torch.cuda.manual_seed_all(cuda_seed)
            torch.backends.cudnn.deterministic = True
            log.info(f'# START: random seeds: torch.cuda {cuda_seed}')

        state_path = None

    else:
        raise ValueError(f'job id {args.jid} is invalid. should be in [1,+] range')

    return state, state_path


def get_state_for_jobid(args, jobid, allow_fail=False):
    '''
    Gets state, state_path for a given jobid
    returns:
      state=None if allow_fail is enabled
    '''

    if (jobid <= 0) or (jobid > args.jid - 2):
        raise ValueError(f'# jobid {jobid} is not valid. it must be in the range [1,args.jid-2] = [1,{args.jid-2}] for args.jid={args.jid}')

    state_path = os.path.join(args.checkpoint_dir, f'state.{jobid}.pkl')
    if allow_fail:
        if not os.path.exists(state_path):
            return None, state_path
    state = load_state(state_path, silent=True)

    return state, state_path


def get_model_from_state(state, model, device):
    ''' Loads a model from a state, mapping it to a given device'''

    if 'paths' in state:  # is state initialized?
        if 'model_path' in state['paths']:
            model.load_state_dict(torch.load(state['paths']['model_path'], map_location=device))
            log.info(f"# Loaded model params from: {state['paths']['model_path']}")
        else:
            raise Exception(f"missing model_path in state file {state['paths']['state_path']}")
    else:
        log.info("# Model is original")

    return model


def get_optimizer_from_state(state, optimizer, device):
    ''' Loads optimizer from a state. mapping it to a given device'''

    if 'paths' in state:  # is state initialized?
        if 'optimizer_path' in state['paths']:
            optimizer.load_state_dict(torch.load(state['paths']['optimizer_path'], map_location=device))
            log.info(f"# Loaded optimizer state from: {state['paths']['optimizer_path']}")
        else:
            raise Exception(f"'missing optimizer_path in state file {state['paths']['state_path']}")
    else:
        log.info("# Optimizer is original")

    return optimizer


def get_args_from_state(state):
    ''' Loads args (argparse) from a state. fails (raise) if it does not exist'''

    args = None
    paths = state.get('paths', None)

    if paths is None:
        raise RuntimeError("provided state has no 'paths' key")

    if 'args_path' in paths:
        args = read_pickle(state['paths']['args_path'])
        log.info(f"# Loaded args state from: {state['paths']['args_path']}")
    else:
        raise Exception(f"missing args_path in state file {state['paths']['state_path']}")

    return args


def checkpoints(args, epoch, iteration, model, optimizer, checkpoint_type='train', last=False):
    ''' Checkpoints model and optimizer for given epoch, iteration'''

    log.info(f"# Checkpointing: epoch {epoch} -- iteration {iteration} [type='{checkpoint_type}']")

    is_train = (checkpoint_type == 'train')
    checkpoint_dir = args.checkpoint_dir if is_train else args.checkpoint_eval_dir

    model_path = os.path.join(checkpoint_dir, f'model.epoch.{epoch}.iter.{iteration}.pth') if is_train \
        else os.path.join(checkpoint_dir, f'model.epoch.{epoch:.7f}.iter.{iteration}.pth')

    log.info(f'# checkpoint model in : {model_path}')
    if args.distributed:
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)

    if checkpoint_type == 'train':
        optimizer_path = os.path.join(checkpoint_dir, f'optimizer.epoch.{epoch}.iter.{iteration}.pth')
        log.info(f'# checkpoint optimizer in : {optimizer_path}')
        torch.save(optimizer.state_dict(), optimizer_path)

    args_path = os.path.join(checkpoint_dir, f'args.epoch.{epoch}.iter.{iteration}.pkl') if is_train \
        else os.path.join(checkpoint_dir, f'args.epoch.{epoch:.7f}.iter.{iteration}.pkl')

    save_pickle(args_path, args, msg='Checkpointing', logger=log)

    state = {}
    state['iteration'] = iteration
    state['epoch'] = epoch
    state['seeds'] = {'torch_state': torch.get_rng_state(),
                      'cuda_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                      'numpy_state': np.random.get_state(),
                      'python_state': random.getstate()}

    state['paths'] = {'model_path': model_path,
                      'optimizer_path': optimizer_path if checkpoint_type == 'train' else None,
                      'state_path': None,  # place holder filled in later below
                      'args_path': args_path
                      }

    state['vars'] = {'model': args.model}  # save some arg vars within the state (like model codename)

    if checkpoint_type == 'train':
        state_pkl = f'state.{args.jid}.epoch.{epoch}.iter.{iteration}.pkl' if not last else f'state.{args.jid}.pkl'
    else:
        # epoch is a "float" number which in Python are double precision  53 bits (16 digits)
        # -> we save the name of the state w/ 7 digits when saving within an epoch.
        state_pkl = f'state.{args.jid}.epoch.{epoch:.7f}.iter.{iteration}.pkl'

    state_path = os.path.join(checkpoint_dir, state_pkl)

    state['paths']['state_path'] = state_path  # save state path within

    log.info(f"# state['paths']: {vars2str(state['paths'])}")
    save_pickle(state_path, state, logger=log)

    return state_path
