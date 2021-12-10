''' Utilities
'''

from typing import Sequence, Optional, Any, Dict  # List
import errno
import json
import os
import pickle
import sys
import time
from xml.etree import ElementTree as ET
from xml.dom import minidom

import logging

logger = logging.getLogger(__name__)
log = logger


class ddict(object):
    ''' Simple dot dictionary '''
    def __init__(self, d):
        self.__dict__ = d


def noop(*args, **kwargs):
    ''' No-op operation '''
    pass


def vars2str(args):
    ''' Given args, returns a string representation.
       Output is either str(args) for non dictionary input or JSON string for dictionary input
    '''
    if not hasattr(args, '__dict__'):
        return str(args)
    return json.dumps(vars(args), sort_keys=True, indent=2)


def mkdir_p(path: str,
            logger: Optional[Any] = None) -> None:
    ''' Replicates the behavior of `mkdir -P` on UNIX.

    Properties:
    - can handle filesystem with unreliable or delayed I/O by attempting dir creation
      multiple time before raising an exception.
    - can handle racing condition of multiple jobs creating the same path at the same time.
    path is canonized with os.path.realpath() before given to os.makedirs.
    '''
    mprint = logger.info if (logger is not None) else print

    if not path:
        mprint('# mkdir_p: missing dir path')
        raise RuntimeError('missing dir path')

    # python doc for  os.makedirs states that:
    # '''Note: makedirs() will become confused if the path elements to create include pardir
    #    (eg. “..” on UNIX systems).'''
    # -> We canonize the path
    rpath = os.path.realpath(path)

    if os.path.exists(rpath) and os.path.isdir(rpath):
        mprint(f'# mkdir_p: dir exists: {path} [{rpath}]')
        return

    attemptN = 3
    for attempt in range(attemptN):
        try:
            os.makedirs(rpath)

        except OSError as exc:  # Python > 2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                mprint(f'# created {path} [attempt= {attempt}]')
                return
            else:
                mprint(f'# dir creation failed: {path} [{rpath}] [attempt={attempt}]')
                time.sleep(1)
                continue
    raise RuntimeError(f'# dir creation failed: {path} [{rpath}] [attempts={attemptN}]')


def mkdirs(paths: Sequence[str],
           logger: Optional[Any] = None) -> None:
    ''' Creates all dirs in paths '''
    for path in paths:
        mkdir_p(path, logger=logger)


def file_exists(path: str,
                logger: Optional[Any] = None,
                silent: bool = False) -> bool:
    ''' File existence check '''
    mprint = logger.info if (logger is not None) else print
    mprint = noop if silent else mprint

    if os.path.isfile(path):
        mprint(f'# {path} exists')
        return True

    return False


PICKLE_DEFAULT_PROTOCOL = 3  # force protocol 3 to use w/ python <3.8


def read_pickle(fname: str,
                msg: str = '',
                logger: Optional[Any] = None,
                silent: bool = False) -> Any:
    ''' Wrapper for reading pickle files '''

    mprint = logger.info if (logger is not None) else print
    mprint = noop if silent else mprint

    msg = msg + ' ' if msg else msg

    with open(fname, 'rb') as fp:
        mprint("# {}Reading {}".format(msg, fname))
        obj = pickle.load(fp)
        if isinstance(obj, dict):
            mprint("# {}Loaded {}  [{} records]".format(msg, fname, len(obj)))
        else:
            mprint("# {}Loaded {}".format(msg, fname))
        return obj

    raise Exception("cannot open for reading {}".format(fname))


def save_pickle(fname: str,
                obj: Any,
                msg: str = '',
                logger: Optional[Any] = None,
                silent: bool = False) -> None:
    ''' Wrapper for saving pickled object to file '''

    mprint = logger.info if (logger is not None) else print
    mprint = noop if silent else mprint

    msg = msg + ' ' if msg else msg

    with open(fname, 'wb') as fp:
        mprint("# {}Saving {}".format(msg, fname))
        pickle.dump(obj, fp, protocol=PICKLE_DEFAULT_PROTOCOL)
        if isinstance(obj, dict):
            mprint("# {}Saved {}  [{} records]".format(msg, fname, len(obj)))
        else:
            mprint("# {}Saved {}".format(msg, fname))
        return

    raise Exception("cannot open for saving {}".format(fname))


def save_json(fname: str,
              obj: Any,
              msg: str = '',
              indent: int = 0,
              sort_keys: bool = False,
              logger: Any = None,
              silent: bool = False) -> None:
    ''' Wrapper for saving JSON object to file '''

    mprint = logger.info if (logger is not None) else print
    mprint = noop if silent else mprint

    msg = msg + ' ' if msg else msg

    try:
        fp = open(fname, 'w')
    except IOError:
        raise Exception("# cannot open file when saving {}".format(fname))
    else:
        with fp:
            mprint("# {}Saving {}".format(msg, fname))
            json.dump(obj, fp, indent=indent, sort_keys=sort_keys)
            if isinstance(obj, dict) or isinstance(obj, list):
                mprint("# {}Saved {}  [{} records]".format(msg, fname, len(obj)))
            else:
                mprint("# {}Saved {}".format(msg, fname))
        return


def read_json(fname: str,
              msg: str = '',
              logger: Any = None,
              silent: bool = False) -> Any:
    ''' Wrapper to read JSON objects '''
    mprint = logger.info if (logger is not None) else print
    mprint = noop if silent else mprint

    msg = msg + ' ' if msg else msg

    try:
        fp = open(fname, 'r')
    except IOError:
        raise Exception("# cannot open for loading {}".format(fname))
    else:
        with fp:
            mprint("# {}Loading {}".format(msg, fname))
            obj = json.load(fp)
            if isinstance(obj, dict) or isinstance(obj, list):
                mprint("# {}Loaded {}  [{} records]".format(msg, fname, len(obj)))
            else:
                mprint("# {}Loaded {}".format(msg, fname))
        return obj


def write_dict_as_list(dataD, fname, msg, sep=' ', what='k+v', logger=None):
    ''' Writes a dict as a list in file '''
    mprint = logger.info if (logger is not None) else print

    mprint(f'# [{len(dataD)} records] : saving {msg} : {fname}')
    with open(fname, 'w') as fp:
        for k, v in dataD.items():
            if what == 'k':
                fp.write(f'{k}\n')
            elif what == 'v':
                fp.write(f'{v}\n')
            else:
                fp.write(f'{k}{sep}{v}\n')


def write_list(dataL, fname, msg, what='i+v', logger=None):
    ''' Writes a list to file '''
    mprint = logger.info if (logger is not None) else print

    mprint(f'# [{len(dataL)} records] : saving {msg} : {fname}')
    with open(fname, 'w') as fp:
        for ii, entry in enumerate(dataL):
            if what == 'i+v':
                fp.write(f'{ii} {entry}\n')
            else:
                fp.write(f'{entry}\n')


def str2bool(val):
    ''' Converts 'val' into a boolean
        if 'val' is a boolean, returns 'val'
        if 'val' is a string, converts the string representation to True or False
        - returns True  for 'yes', 'y', 'true',  't', 'on',  '1'):
        - returns False for 'no',  'n', 'false', 'f', 'off', '0'):
        Raise ValuerError if 'val' is not a boolean or a string
    '''
    if isinstance(val, bool):
        return val
    elif isinstance(val, str):
        v = val.lower()
        if v in ('yes', 'y', 'true', 't', 'on', '1'):
            return True
        elif v in ('no', 'n', 'false', 'f', 'off', '0'):
            return False
        else:
            raise ValueError('string representation of a boolean value expected.')
    else:
        raise ValueError(f"input 'val' must be a bool, a string. not {type(val)}")


def count_model_params(model):
    ''' Returns the total number of parameters and total number of trainable parameters'''
    params_num = sum(p.numel() for p in model.parameters())
    train_params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num, train_params_num


def get_cuda_visible_devices() -> Dict[str, str]:
    ''' Returns CUDA devices from ENV'''
    envD = os.environ
    cuda_env_vars = ['FULL_CUDA_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES_ORIG', 'CUDA_VISIBLE_DEVICES']

    visible_devices = {}
    for env_var in cuda_env_vars:
        if env_var in envD:
            visible_devices[env_var] = envD[env_var]

    return visible_devices


def log_cuda_visible_devices(logger=None) -> Dict[str, str]:
    ''' Logs CUDA devices from ENV'''
    mprint = logger.info if (logger is not None) else print

    cuda_visible_devices = get_cuda_visible_devices()

    for k, v in cuda_visible_devices.items():
        mprint(f'# {k}= {v}')

    return cuda_visible_devices


def log_versions(logger=None, modules=None):
    ''' Logs versions of modules '''
    mprint = logger.info if (logger is not None) else print

    # python
    mprint(f"# versions: python : {sys.version}")

    if modules is None:
        return

    # from https://stackoverflow.com/questions/36283541/how-to-check-version-of-python-package-if-no-version-variable-is-set
    from pip._vendor import pkg_resources

    for package in modules:
        package = package.lower()
        ver = next((p.version for p in pkg_resources.working_set if p.project_name.lower() == package), "no match")

        mprint(f'# version: {package} : {ver}')

    return


def save_xml(fname: str,
             obj: Any,
             msg: str = '',
             indent: int = 0,
             sort_keys: bool = False,
             logger: Any = None,
             silent: bool = False) -> None:
    ''' Wrapper to save object to XML '''

    mprint = logger.info if (logger is not None) else print
    mprint = noop if silent else mprint
    msg = msg + ' ' if msg else msg

    tree = obj
    root = tree.getroot()
    et_string = ET.tostring(root, 'utf-8')  # you need to start from root node
    reparsed = minidom.parseString(et_string)
    out_string = reparsed.toprettyxml(indent="  ")

    root = tree.getroot()
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(out_string)

    mprint(f"# {msg}Saved {fname} [{len(root[0])} records]")

    return


def read_xml(fname: str,
             msg: str = '',
             logger: Any = None,
             silent: bool = False) -> Any:
    ''' Wrapper to read object from XML '''
    mprint = logger.info if (logger is not None) else print
    mprint = noop if silent else mprint
    msg = msg + ' ' if msg else msg

    tree = ET.parse(fname)
    root = tree.getroot()
    mprint(f"# {msg}Loaded {fname} [{len(root[0])} records]")

    return tree
