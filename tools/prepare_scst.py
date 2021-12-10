""" Prepare stage for SCST training
"""
import logging.handlers
import logging
import configargparse
import os
import shutil

import opts

from utils.state import load_state
from utils.utils import mkdirs


# ------------------------------
# Logger
# ------------------------------
# logger
logger = logging.getLogger()  # get root logger
logger.setLevel(logging.DEBUG)
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
        # base = os.path.basename(__file__)
        logdir = args.log_dir
        mkdirs([logdir], logger=log)
        log_filename = os.path.join(logdir, 'prepare_scst.log')
        file_handler = logging.FileHandler(log_filename, mode='w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        # add handler
        logger.addHandler(file_handler)
        log.info('# File logger started')

    if args.scst_checkpoint_id <= 0:
        raise ValueError(f"checkpoint id '{args.scst_checkpoint_id}' must be defined (and greater than 0)")

    state_path = os.path.join(args.scst_checkpoint_dir, f'state.{args.scst_checkpoint_id}.pkl')

    # exists?
    if not os.path.exists(state_path):
        raise ValueError(f"# state file '{state_path}' does not exist")

    state = load_state(state_path, rebase_paths=True)

    # destination dir
    dest_dir = os.path.join(args.output_dir, 'checkpoints')
    mkdirs([dest_dir], logger=log)

    for fname, fpath in state['paths'].items():

        if not os.path.exists(fpath):
            log.info(f"# '{fname}':  file '{fpath}' does not exist -> skip")
        else:
            fbase = os.path.basename(fpath)
            dest = os.path.join(dest_dir, fbase)
            if args.mode == 'copy':
                shutil.copy2(fpath, dest)
            elif args.mode == 'symlink':
                os.symlink(fpath, dest)
            log.info(f'# {fpath} -> {dest} [{args.mode}]')

    log.info("# done")


if __name__ == "__main__":

    # Parsing arguments
    p = configargparse.ArgParser(description='config file arguments')

    # require config file
    p.add("-c", "--config", required=True, is_config_file=True, help="config file path")
    p.add("-m", "--mode", required=True, choices=['copy', 'symlink'], help="mode for transfer of CE models: [copy, symlink]")

    p = opts.add_arguments(p)

    args = p.parse_args()

    log.info(p.format_values())

    # check arguments, add some too
    args = opts.check_and_add_arguments(args)

    main(args)
