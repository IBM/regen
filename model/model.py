import logging

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
import cfg

logger = logging.getLogger(__name__)
log = logger

CACHE_DIR = cfg.TRANSFORMERS_CACHE


def get_model_name(model_codename):
    ''' Returns a (full) model name from a (simple) model codename '''

    model_name = cfg.MODEL_MAP.get(model_codename, None)

    if model_name is None:
        raise RuntimeError(f"model codename '{model_codename}' does not correspong to a valid model as in {cfg.MODEL_LIST}")

    return model_name


def get_model_and_tokenizer_classes(model_codename):
    ''' Returns model and tokenizer classes from model codename'''

    model_name = get_model_name(model_codename)

    if model_name is None:
        raise RuntimeError(f"model codename '{model_codename}' does not correspong to a valid model as in {cfg.MODEL_LIST}")

    if model_name in ['facebook/bart-base', 'facebook/bart-large']:
        tokenizer_name = 'bart'
        tokenizer_class = BartTokenizer
        model_class = BartForConditionalGeneration

    elif model_name in ['t5', 't5-small', 't5-base', 't5-large', 't5-3b', 't5-11b']:
        tokenizer_name = 't5'
        tokenizer_class = T5Tokenizer
        model_class = T5ForConditionalGeneration

    elif model_name in ['new']:
        tokenizer_name = 'unk'
        tokenizer_class = None
        model_class = None

    else:
        raise RuntimeError(f"model '{model_name}' is not valid")

    return model_name, model_class, tokenizer_name, tokenizer_class


def get_model(model_codename):
    ''' Returns a model (instance) from a model codename'''

    model_name, model_class, _, _ = get_model_and_tokenizer_classes(model_codename)

    if model_class is not None:
        log.info(f"# Loading model '{model_codename}': '{model_name}'")
        model = model_class.from_pretrained(model_name, cache_dir=CACHE_DIR)
    else:
        raise ValueError(f"model_class is not defined for '{model_name}'")

    return model, model_name


def get_tokenizer(model_codename):
    ''' Returns a tokenizer (instance) from a model codename'''
    model_name, _, tokenizer_name, tokenizer_class = get_model_and_tokenizer_classes(model_codename)

    if tokenizer_class is not None:
        log.info(f"# Loading tokenizer '{model_codename}':  model: '{model_name}'  tokenizer: '{tokenizer_name}'")
        tokenizer = tokenizer_class.from_pretrained(model_name, cache_dir=CACHE_DIR)
    else:
        raise ValueError(f"'tokenizer_class is not defined for '{model_name}'")

    return tokenizer, tokenizer_name


def get_model_and_tokenizer(model_codename):
    ''' Returns both model and tokenizer'''
    model, _ = get_model(model_codename)
    tokenizer, _ = get_tokenizer(model_codename)

    return model, tokenizer


def get_tokenizer_name(tokenizer):
    ''' Returns a tokenizer name from a tokenizer '''

    tokenizer_name = None
    if isinstance(tokenizer, T5Tokenizer):
        tokenizer_name = 't5'
    elif isinstance(tokenizer, BartTokenizer):
        tokenizer_name = 'bart'
    else:
        tokenizer_name = None

    return tokenizer_name
