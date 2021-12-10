import numpy as np

from utils.metrics import (
    parse,
    bleu_nltk,
    meteor_nltk,
    sacre_bleu,
    sacre_chrf,
    sacre_ter,
    webnlg_rdf_exactF1
)

from utils.utils_webnlg_parse import get_triples
from utils.utils_tekgen_parse import get_tekgen_triples

import logging

logger = logging.getLogger(__name__)
log = logger


def self_critical_reward_triple(greedy_seq, target_seq, sample_seq, args, normalize_weights=True):
    '''
    Computes SCST reward for triple given greedy decode results (greedy_seq), ground_truth (target_seq)
    and generated sample results (sample_seq)
    '''

    bsize = len(target_seq)

    webnlg_exact_F1_weight = args.webnlg_exact_F1_weight if args.webnlg_exact_F1_weight is not None else 0.0
    bleu_nltk_weight = args.bleu_nltk_weight if args.bleu_nltk_weight is not None else 0.0

    webnlg_exact_F1_reward = 0.0
    bleu_nltk_reward = 0.0

    rewards = {}
    weights = []
    scores = []

    if args.webnlg_exact_F1_weight is not None:

        parse_triples = get_tekgen_triples if args.prepare == "tekgen-official" else get_triples

        greedy_seq_fmt = parse_triples(greedy_seq)
        target_seq_fmt = parse_triples(target_seq)
        sample_seq_fmt = parse_triples(sample_seq)

        score_sample = webnlg_rdf_exactF1(target_seq_fmt, sample_seq_fmt)  # check
        score_greedy = webnlg_rdf_exactF1(target_seq_fmt, greedy_seq_fmt)

        rewards['webnlg_exact_F1_sample'] = score_sample
        rewards['webnlg_exact_F1_greedy'] = score_greedy

        webnlg_exact_F1_reward = score_sample - score_greedy

        weights.append(webnlg_exact_F1_weight)
        scores.append(webnlg_exact_F1_reward)

    if args.bleu_nltk_weight is not None:

        # parse sequences as sentences
        greedy_seq_tok = parse(greedy_seq)
        target_seq_tok = parse(target_seq)
        sample_seq_tok = parse(sample_seq)

        bleu_nltk_sample = bleu_nltk(target_seq_tok, sample_seq_tok)  # reward(w^s) :  sample w^s from model
        bleu_nltk_greedy = bleu_nltk(target_seq_tok, greedy_seq_tok)  # reward(\hat(w)) : test time inference

        rewards['bleu_nltk_sample'] = bleu_nltk_sample
        rewards['bleu_nltk_greedy'] = bleu_nltk_greedy

        bleu_nltk_reward = bleu_nltk_sample - bleu_nltk_greedy   # SCST reward: r(w^s) - r(\hat(w)

        weights.append(bleu_nltk_weight)
        scores.append(bleu_nltk_reward)

    # reward
    reward = np.zeros((bsize), np.float32)
    norm = sum(weights) if normalize_weights else 1.0

    for weight, score in zip(weights, scores):
        reward += weight / norm * score

    return {'reward': reward, 'rewards': rewards}


def self_critical_reward_text(greedy_seq, target_seq, sample_seq, args, normalize_weights=True):
    '''
    Computes SCST reward for text given greedy decode results (greedy_seq), ground_truth (target_seq)
    and generated sample results (sample_seq)
    '''
    bsize = len(target_seq)

    bleu_nltk_weight = args.bleu_nltk_weight if args.bleu_nltk_weight is not None else 0.0
    meteor_nltk_weight = args.meteor_nltk_weight if args.meteor_nltk_weight is not None else 0.0

    sacre_bleu_weight = args.sacre_bleu_weight if args.sacre_bleu_weight is not None else 0.0
    sacre_chrf_weight = args.sacre_chrf_weight if args.sacre_chrf_weight is not None else 0.0
    sacre_ter_weight = args.sacre_ter_weight if args.sacre_ter_weight is not None else 0.0

    bleu_nltk_reward = 0.0
    meteor_nltk_reward = 0.0

    sacre_bleu_reward = 0.0
    sacre_chrf_reward = 0.0
    sacre_ter_reward = 0.0

    greedy_seq_tok = parse(greedy_seq)
    target_seq_tok = parse(target_seq)
    sample_seq_tok = parse(sample_seq)

    rewards = {}
    weights = []
    scores = []

    if args.sacre_bleu_weight is not None:

        sacre_bleu_sample = sacre_bleu(target_seq_tok, sample_seq_tok)  # check
        sacre_bleu_greedy = sacre_bleu(target_seq_tok, greedy_seq_tok)

        rewards['sacre_bleu_sample'] = sacre_bleu_sample
        rewards['sacre_bleu_greedy'] = sacre_bleu_greedy

        sacre_bleu_reward = sacre_bleu_sample - sacre_bleu_greedy

        weights.append(sacre_bleu_weight)
        scores.append(sacre_bleu_reward)

    if args.sacre_chrf_weight is not None:

        sacre_chrf_sample = sacre_chrf(target_seq_tok, sample_seq_tok)  # check
        sacre_chrf_greedy = sacre_chrf(target_seq_tok, greedy_seq_tok)

        rewards['sacre_chrf_sample'] = sacre_chrf_sample
        rewards['sacre_chrf_greedy'] = sacre_chrf_greedy

        sacre_chrf_reward = sacre_chrf_sample - sacre_chrf_greedy

        weights.append(sacre_chrf_weight)
        scores.append(sacre_chrf_reward)

    if args.sacre_ter_weight is not None:

        sacre_ter_sample = sacre_ter(target_seq_tok, sample_seq_tok)  # check
        sacre_ter_greedy = sacre_ter(target_seq_tok, greedy_seq_tok)

        rewards['sacre_ter_sample'] = sacre_ter_sample
        rewards['sacre_ter_greedy'] = sacre_ter_greedy

        sacre_ter_reward = sacre_ter_sample - sacre_ter_greedy

        weights.append(sacre_ter_weight)
        scores.append(sacre_ter_reward)

    if args.bleu_nltk_weight is not None:

        bleu_nltk_sample = bleu_nltk(target_seq_tok, sample_seq_tok)  # reward(w^s) :  sample w^s from model
        bleu_nltk_greedy = bleu_nltk(target_seq_tok, greedy_seq_tok)  # reward(\hat(w)) : test time inference

        rewards['bleu_nltk_sample'] = bleu_nltk_sample
        rewards['bleu_nltk_greedy'] = bleu_nltk_greedy

        bleu_nltk_reward = bleu_nltk_sample - bleu_nltk_greedy   # SCST reward: r(w^s) - r(\hat(w)

        weights.append(bleu_nltk_weight)
        scores.append(bleu_nltk_reward)

    if args.meteor_nltk_weight is not None:

        meteor_nltk_sample = meteor_nltk(target_seq_tok, sample_seq_tok)  # reward(w^s) :  sample w^s from model
        meteor_nltk_greedy = meteor_nltk(target_seq_tok, greedy_seq_tok)  # reward(\hat(w)) : test time inference

        rewards['meteor_nltk_sample'] = meteor_nltk_sample
        rewards['meteor_nltk_greedy'] = meteor_nltk_greedy

        meteor_nltk_reward = meteor_nltk_sample - meteor_nltk_greedy   # SCST reward: r(w^s) - r(\hat(w)

        weights.append(meteor_nltk_weight)
        scores.append(meteor_nltk_reward)

    # reward
    reward = np.zeros((bsize), np.float32)
    norm = sum(weights) if normalize_weights else 1.0

    for weight, score in zip(weights, scores):
        reward += weight / norm * score

    return {'reward': reward, 'rewards': rewards}


def self_critical_reward_AB(greedy_seq, target_seq, sample_seq, args):
    '''
    Computes reward for mixed AB training [A+B] w/ text and graph samples (not implemented)
    '''
    reward = 0.0
    log.warning('not implemented')
    return reward
