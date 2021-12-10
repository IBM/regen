import os
from addict import Dict as adict

# Paths to datasets
# all paths are relative to {root_dir}
root_dir = os.path.dirname(os.path.abspath(__file__))

# WebNLG
datasets_webnlg_path = f'{root_dir}/corpora/webnlg-dataset'
datasets_webnlg_meta_path = '{root_dir}/corpora/webnlg-references'

# TekGen
datasets_tekgen_official_path = f'{root_dir}/corpora/tekgen/official'
datasets_tekgen_default_path = f'{root_dir}/corpora/tekgen/prepare.v1.0'

# Huggingface caches paths -- adjust to your filesystem
TRANSFORMERS_CACHE = f'{root_dir}/caches/huggingface/models_cache/'
HF_DATASETS_CACHE = f'{root_dir}/caches/huggingface/datasets_cache/'
HF_METRICS_CACHE = f'{root_dir}/caches/huggingface/metrics_cache/'

# ---
dataset_choices = ['webnlg', 'tekgen']

# Data splits
# Notation convention:  A=text, B=graph
# so
#
# 'testA' means test for t2g task
# 'valA'  means val  for t2g task
#
# 'testB' means test for g2t task
# 'valB'  means val  for g2t task
#
#  source: A (text)  implies T2G (AB) task
#  source: B (graph) implies G2T (BA) task
#  source: A+B (text+graph) can allow for both generation tasks (G2T/T2G)

split_choices = ['val', 'testA', 'testB']  # for evaluation
split_choices_all = ['train', 'val', 'testA', 'testB']  # for evaluation + training
split_logic_choices = ['valA', 'valB', 'testA', 'testB']  # for evaluation
src_choices = ['A', 'B', 'A+B']

# Special tokens
# Tokens for graph edge (head, relops, tail) = (subject, predicate, object)
SPECIAL_TOKENS_DICT = adict()
SPECIAL_TOKENS_DICT.subj = '__subject__'
SPECIAL_TOKENS_DICT.pred = '__predicate__'
SPECIAL_TOKENS_DICT.obj = '__object__'

SPECIAL_TOKENS_WEBNLG = [
    SPECIAL_TOKENS_DICT.subj,
    SPECIAL_TOKENS_DICT.pred,
    SPECIAL_TOKENS_DICT.obj]

SPECIAL_TOKENS_TEKGEN = [
    SPECIAL_TOKENS_DICT.subj,
    SPECIAL_TOKENS_DICT.pred,
    SPECIAL_TOKENS_DICT.obj]

# Tasks prefix
task_prefix_A = 'Text to Graph:'  # t2g (AB) task prefix (do *not* change)
task_prefix_B = 'Graph to Text:'  # g2t (BA) task prefix (do *not* change)

# Preparation engines (not necessarily the same as dataset_choices)
# A preparation engine formats the text and graph data
PREPARE_ENGINES = ['webnlg',
                   'webnlg-lex1',
                   'webnlg-lex1-kbpm',
                   'tekgen-default',
                   'tekgen-official']

# Mapping between model simple 'codenames' and HuggingFace (HF) model names
MODEL_MAP = {
    'bart-base': 'facebook/bart-base',
    'bart-large': 'facebook/bart-large',
    't5': 't5',
    't5-small': 't5-small',
    't5-base': 't5-base',
    't5-large': 't5-large',
    't5-3b': 't5-3b',
    't5-11b': 't5-11b',
    'new': 'new'
}

MODEL_LIST = list(MODEL_MAP.keys())

# Options for data permutation
permutation_choices = ['none', 'shuffle', 'internal']

# WebNLG has multiple subsets of data
webnlg_subsets = ['all', 'seen', 'unseen']

# Generation
# Datasets and prepare engines allowed for generation:
GENERATION_DATASETS = ['webnlg', 'tekgen']
GENERATION_PREPARE = ['webnlg', 'tekgen-official']
