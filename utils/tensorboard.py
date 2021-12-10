from collections import OrderedDict
import glob
import json
import shutil
import os
import logging
import torch

logger = logging.getLogger(__name__)
log = logger


class LoggingWriter(object):
    ''' Provides an alternative to TensorBoard using python logging infrastructure.'''
    def __init__(self, writer_path='', writer_type='log'):

        self.writer_path = writer_path
        self.writer_type = writer_type

    def add_anything(self, type_name, key, value, global_step, walltime=None):
        log.info(f'"writer": {self.writer_type} "type_name": {type_name}, "global_step": {global_step}, "walltime": {walltime}, "key": {key}, "value": {value}')

    def add_scalar(self, key, value, global_step=None, walltime=None):
        self.add_anything('scalar', key, value, global_step, walltime)

    def add_text(self, key, value, global_step=None, walltime=None):
        self.add_anything('text', key, value, global_step, walltime)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        self.add_dict(main_tag, tag_scalar_dict, global_step, walltime, type_name='scalars')

    def add_dict(self, main_tag, val_dict, global_step=None, walltime=None, type_name='dict'):
        self.add_anything(type_name, main_tag, val_dict, global_step, walltime)

    def add_text_list(self, key, value, global_step=None, walltime=None):
        self.add_anything('text_list', key, value, global_step, walltime)


class JsonLoggingWriter(object):
    ''' Provides an alternative to TensorBoard using python-json-logger library.'''
    def __init__(self, json_logger, writer_log_file=''):

        self.json_log = logging.getLogger('json-log')
        self.json_log.propagate = False

        # format for json logger
        format_str = '%(asctime)%(created)%(levelname)%(message)'

        # file
        log_filename = writer_log_file
        file_handler = logging.FileHandler(log_filename, mode='w')
        formatter = json_logger.JsonFormatter(format_str)
        file_handler.setFormatter(formatter)
        self.json_log.addHandler(file_handler)

        self.log = self.json_log

    def add_anything(self, type_name, key, value, global_step, walltime=None):
        self.log.info(f'{type_name}', extra={'global_step': global_step, "walltime": walltime, key: value})

    def add_scalar(self, key, value, global_step=None, walltime=None):
        self.add_anything('scalar', key, value, global_step, walltime)

    def add_text(self, key, value, global_step=None, walltime=None):
        self.add_anything('text', key, value, global_step, walltime)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        self.add_dict(main_tag, tag_scalar_dict, global_step, walltime, type_name='scalars')

    def add_dict(self, main_tag, val_dict, global_step=None, walltime=None, type_name='dict'):
        self.log.info(type_name, extra={'iteration': global_step, main_tag: val_dict})

    def add_text_list(self, key, value, global_step=None, walltime=None):
        self.add_anything('text_list', key, value, global_step, walltime)


class TbWriter(object):
    ''' Provides a lightweight  wrapper to TensorBoard.'''
    def __init__(self, tb, tb_path, purge_step, remove_past_events=True):

        # remove previous events files
        if remove_past_events and os.path.exists(tb_path):
            log.info(f'# clean {tb_path}')
            tb_path_bak = tb_path + '.bak'
            shutil.move(tb_path, tb_path_bak)
            shutil.rmtree(tb_path_bak)
            os.makedirs(tb_path)

        self.tb = tb.SummaryWriter(tb_path, purge_step=purge_step)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        self.tb.add_scalar(tag, scalar_value, global_step, walltime)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        self.tb.add_text(tag, text_string, global_step, walltime)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        self.tb.add_dict(main_tag, tag_scalar_dict, global_step, walltime)

    def add_dict(self, main_tag, val_dict, global_step=None, walltime=None):
        pass

    def add_text_list(self, tag, text_list, global_step=None, walltime=None):
        val_str = '<br/>'.join(text_list)
        val_str += '<br/>'
        self.tb.add_text(tag, val_str, global_step, walltime)


class Writer(object):
    ''' Writer class to log to TensorBoard, SummaryWriter, and JSON logger.'''

    def __init__(self, tb, json_logger, tb_path, json_file_path, purge_step, logger=None):

        # local logging
        self.log = logger if logger is not None else log

        # tb writer
        if tb is not None:
            self.tb_type = 'tb'
            self.tb_writer = TbWriter(tb, tb_path, purge_step, remove_past_events=True)
        else:
            self.tb_type = 'log'
            self.tb_writer = LoggingWriter(writer_path=tb_path, writer_type='tb')  # provide logging writer as default

        # json writer
        self.jsonlog_type = 'log'
        if json_logger is not None:
            self.json_writer = JsonLoggingWriter(json_logger, writer_log_file=json_file_path)
        else:
            self.json_writer = LoggingWriter(writer_path=json_file_path, writer_type='json')  # provide logging writer as default

        self.writers = [self.tb_writer, self.json_writer]

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        if isinstance(scalar_value, torch.Tensor):
            scalar_value = scalar_value.item()
        for writer in self.writers:
            writer.add_scalar(tag, scalar_value, global_step, walltime)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        for writer in self.writers:
            writer.add_text(tag, text_string, global_step, walltime)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        for writer in self.writers:
            writer.add_scalars(main_tag, tag_scalar_dict, global_step, walltime)

    def add_dict(self, main_tag, val_dict, global_step=None, walltime=None):
        '''Saves a dict w/ simple value type as scalar, string, dict.
        This call is *not* in TensorBoard API, it is a no op for a TensorBoard writer
        '''
        for writer in self.writers:
            writer.add_dict(main_tag, val_dict, global_step, walltime)

    def add_text_list(self, tag, text_list, global_step=None, walltime=None):
        for writer in self.writers:
            writer.add_text_list(tag, text_list, global_step, walltime)


def get_summary_writer(tb_path='', json_file_path='', global_iteration=0, logger=None):
    ''' Returns a summary writer with tensorboard, JSON logger if available '''
    if logger is not None:
        log = logger
    else:
        logger = logging.getLogger(__name__)
        log = logger

    # TensorBoard ?
    try:
        import torch.utils.tensorboard as tb
        log.info("# module tensorboard exists")
    except ImportError:
        log.info("#  missing module tensorboard")
        tb = None

    # JSON logger?
    try:
        from pythonjsonlogger import jsonlogger as json_logger
        log.info("# module python-json-logger exists")
    except ImportError:
        log.info("# missing module python-json-logger")
        json_logger = None

    log.warning('# OVERRIDE: force global_iteration/purge_step to be None')
    global_iteration = None

    writer = Writer(tb, json_logger, tb_path, json_file_path, global_iteration, logger=log)

    return writer


def get_tensorboard(args, iteration, logger):
    ''' Returns a "tensorboard" writer '''
    return get_summary_writer(tb_path=args.tensorboard_dir, json_file_path=args.results_json_log, global_iteration=iteration, logger=logger)


def read_json_logs(fname, quiet=True):
    ''' Reads json logs and returns a list of objects '''

    loglines = []
    with open(fname, 'r') as f:
        lines = f.read().splitlines()

    for idx, line in enumerate(lines):
        # log.info(line)

        try:
            obj = json.loads(line)
            loglines.append(obj)
        except json.JSONDecodeError as err:
            log.info(f"# file '{fname}': line {idx} raises an exception by json decoder\n: line= {line}")
            raise RuntimeError(f"Failed parsing '{fname}'") from err

    if not quiet:
        log.info(f'# read fname [{len(line)} lines]')

    return loglines


def get_json_logs(output_dir):
    ''' Given an output_dir for evaluation, collects all jsonlog results files '''

    pathsL = sorted(glob.glob(f'{output_dir}/*/jsonlog/results.json.log'))

    if not pathsL:
        raise RuntimeError(f'cannot find json log files in [{output_dir}]')

    # sort by epoch
    filesD = {}
    for path in pathsL:
        dirname1 = os.path.dirname(path)  # {output_dir}/01/jsonlog
        dirname2 = os.path.dirname(dirname1)  # {output_dir}/01
        job_id = os.path.basename(dirname2)  # 01
        job_num = float(job_id)
        filesD[job_num] = path

    od = OrderedDict(sorted(filesD.items()))

    return od


def get_json_logs_train(output_dir):
    ''' Given an output_dir for training, collects all jsonlog results files '''

    pathsL = sorted(glob.glob(f'{output_dir}/jsonlog/*/results.json.*.rank.0.log'))

    if not pathsL:
        raise RuntimeError(f'cannot find json log files in [{output_dir}]')
    # sort by epoch
    filesD = {}
    for path in pathsL:
        dirname1 = os.path.dirname(path)  # {output_dir}/jsonlog/1
        job_id = os.path.basename(dirname1)  # 1
        job_num = int(job_id)
        filesD[job_num] = path

    od = OrderedDict(sorted(filesD.items()))

    return od
