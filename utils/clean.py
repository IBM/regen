import logging
import os

from utils.state import get_state_for_jobid

logger = logging.getLogger(__name__)
log = logger

valid_targets = ['optimizer']  # allows for more targets in the future
default_targets = ['optimizer']


def cleanup(args, targets=None, remove_files=False, remove_all_past=False, skip=None):
    ''' Cleans up targets from training steps
    Arguments:
    - args : arguments from CLI
    - targets are the names of groups of targets. For now just ['optimizer'] is defined: default is ['optimizer']
    - remove_files : boolean to decide if to truly remove any files or not (to allow for dry-runs)
    - remove_all_past : boolean to decide if you remove *all* past files from begining of training 1 to args.jid - 2
    - skip : list of jobid to skip from deletion no matter what (like for initial state/model copied from somewhere else)
    The target files are determined from args.jid
    - The method will cleanup any files defined in the groups from args.jid - 2
    - We *do not* erase files from the previous iteration (args.jid - 1) to avoid
      to lose needed previous files (optimizer, etc.) in case the current job needs to be restarted.

    '''

    targets = default_targets if targets is None else targets

    for t in targets:
        if t not in valid_targets:
            raise ValueError(f'# target {t} is  not a valid target from {valid_targets}')

    target_id = args.jid - 2  # we target anything 2 jobs prior to avoid issues if we have to restart current training

    if target_id <= 0:
        log.info(f'# cleanup: no targets [target_id={target_id}]')
        return

    if skip is not None:
        if target_id in skip:
            log.info(f'# cleanup: skip [target_id={target_id} in skip={skip}]')
            return

    if remove_all_past:
        for tid in range(1, target_id + 1):
            cleanup_jobid(args, tid, targets, remove_files=remove_files)
    else:
        cleanup_jobid(args, target_id, targets, remove_files=remove_files)

    return


def cleanup_jobid(args, jobid, targets, remove_files=True):
    ''' Cleans up targets for a given jobid
        - jobid and targets must be explicitly given
        - jobid must be in agreement with args.jobid:
          if mut be at most args.jobid-2 to avoid erasing files you may need if you need to restart current training job
    '''

    targets = default_targets if targets is None else targets

    for t in targets:
        if t not in valid_targets:
            raise ValueError(f'# target {t} is not a valid target from {valid_targets}')

    if (jobid <= 0) or (jobid > args.jid - 2):
        raise ValueError(f'# jobid {jobid} is not valid. it must be in the range [1,args.jid-2] = [1,{args.jid-2}] for args.jid={args.jid}')

    state, state_path = get_state_for_jobid(args, jobid, allow_fail=True)

    if state is None:
        log.info(f'# jobid {jobid} state is missing -> skip')
        return

    # is state initialized properly?
    if 'paths' not in state:
        log.warn(f"# jobid {jobid}: state {state_path} does not seem to contain 'paths' key. -> skip")
        return

    # get files to be removed
    removable_files = []
    for t in targets:
        if t == 'optimizer':
            # optimizers
            if 'optimizer_path' in state['paths']:
                optimizer_path = state['paths']['optimizer_path']
                removable_files.append(optimizer_path)
            else:
                log.warn("# missing optimizer_path in state")

    for f in removable_files:
        if os.path.exists(f):
            if remove_files:
                os.remove(f)

            msg = '[removed]' if remove_files else '[do not remove]'
            log.info(f'# jobid {jobid}: {f} exists -> {msg}')
        else:
            log.warn(f'# jobid {jobid}: {f} missing')

    log.info(f'# jobid {jobid} cleaned up')

    return
