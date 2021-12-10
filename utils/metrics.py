import numpy as np
import nltk
import nltk.translate.bleu_score as nltk_bleu_score
import nltk.translate.meteor_score as nltk_meteor_score

import sacrebleu

from eval.webnlg_t2g.Evaluation_script import compute_exactF1

import logging
logger = logging.getLogger(__name__)
log = logger


def parse(sentences):
    ''' Parses sentences '''
    sentences_tok = [' '.join(nltk.word_tokenize(sent)) for sent in sentences]
    return sentences_tok


def bleu_nltk(references, hypotheses):
    ''' Computes NLTK BLEU scores for each reference / hypothesis pairs in references and hypotheses
    Args:
       references: List of reference (assumed to be tokenized)
       hypothesis: List of hypotheses (assumed to be tokenized)
    '''

    if len(references) != len(hypotheses):
        raise RuntimeError(f'references {len(references)} and hypotheses {len(hypotheses)} are lists that should be of same size')

    # references: ["Stanford University is A.T. Charlie Johnson's almaMater. Stanford University is the alma mater for A.T. Charlie Johnson.",
    #              'The Alhambra is 63.8 metres long and has a Humphrys, Tennant and Dykes (of London) power type. The Alhambra, which is 63800.0 mm long, is powered by the London based, Humphrys, Tennant and Dykes engine.']

    # hypotheses: ["Stanford University is A.T. Charlie Johnson's almaMater. University is his almaMer of A.T. Charlie Johnson.",
    #              'The Alhambra, 6380m long and has a Humphrys, Tennant and Dykes (of London) power type. latterhambra is however is 63.80.0 millmms. was  by  Hu based Hu Humphrys, Tennant and Dykes..']

    chencherry = nltk_bleu_score.SmoothingFunction()  # method3 to match webNLG metrics

    scores = []

    for i, reference in enumerate(references):

        ref = reference.strip()
        if not ref:
            raise RuntimeError(f'reference is empty of valid words for reference at index {i}')

        ref_words = ref.split()
        hyp_words = hypotheses[i].split()

        score = nltk_bleu_score.sentence_bleu([ref_words], hyp_words, smoothing_function=chencherry.method3)  # [ref], hyp... [ref] is important!
        scores.append(score)

    scores_np = np.array(scores)

    return scores_np


def meteor_nltk(references, hypotheses):
    ''' Computes NLTK METEOR scores for each reference / hypothesis pairs in references and hypotheses
    Args:
       references: List of reference (assumed to be tokenized)
       hypothesis: List of hypotheses (assumed to be tokenized)
    '''

    if len(references) != len(hypotheses):
        raise RuntimeError(f'references {len(references)} and hypotheses {len(hypotheses)} are lists that should be of same size')

    # references: ["Stanford University is A.T. Charlie Johnson's almaMater. Stanford University is the alma mater for A.T. Charlie Johnson.",
    #              'The Alhambra is 63.8 metres long and has a Humphrys, Tennant and Dykes (of London) power type. The Alhambra, which is 63800.0 mm long, is powered by the London based, Humphrys, Tennant and Dykes engine.']

    # hypotheses: ["Stanford University is A.T. Charlie Johnson's almaMater. University is his almaMer of A.T. Charlie Johnson.",
    #              'The Alhambra, 6380m long and has a Humphrys, Tennant and Dykes (of London) power type. latterhambra is however is 63.80.0 millmms. was  by  Hu based Hu Humphrys, Tennant and Dykes..']

    scores = []

    for i, reference in enumerate(references):

        ref = reference.strip()
        if not ref:
            raise RuntimeError(f'reference is empty of valid words for reference at index {i}')

        hyp = hypotheses[i].strip()

        score = nltk_meteor_score.single_meteor_score(ref, hyp)  # ref, hyp for single_meteor_score
        scores.append(score)

    scores_np = np.array(scores)

    return scores_np


def sacre_bleu(references, hypotheses):
    ''' Computes Sacre BLEU scores for each reference / hypothesis pairs in references and hypotheses
    Args:
       references: List of reference (assumed to be tokenized)
       hypothesis: List of hypotheses (assumed to be tokenized)
    '''

    if len(references) != len(hypotheses):
        raise RuntimeError(f'references {len(references)} and hypotheses {len(hypotheses)} are lists that should be of same size')

    scores = []

    for i, reference in enumerate(references):

        ref = reference.strip()
        if not ref:
            raise RuntimeError(f'reference is empty of valid words for reference at index {i}')

        ref_ = ref
        hyp_ = hypotheses[i]

        bleu = sacrebleu.sentence_bleu(hyp_, [ref_])  # hyp, [ref] : order is different than nltk and list '[ref]' is important!
        scores.append(bleu.score / 100.0)

    scores_np = np.array(scores)

    return scores_np


def sacre_chrf(references, hypotheses):
    ''' Computes Sacre CHRF scores for each reference / hypothesis pairs in references and hypotheses
    Args:
       references: List of reference (assumed to be tokenized)
       hypothesis: List of hypotheses (assumed to be tokenized)
    '''

    if len(references) != len(hypotheses):
        raise RuntimeError(f'references {len(references)} and hypotheses {len(hypotheses)} are lists that should be of same size')

    scores = []

    for i, reference in enumerate(references):

        ref = reference.strip()
        if not ref:
            raise RuntimeError(f'reference is empty of valid words for reference at index {i}')

        ref_ = ref
        hyp_ = hypotheses[i]

        chrf = sacrebleu.sentence_chrf(hyp_, [ref_])  # hyp, [ref]: in this order and list '[ref]' is important!
        scores.append(chrf.score)

    scores_np = np.array(scores)

    return scores_np


def sacre_ter(references, hypotheses):
    ''' Computes Sacre TER scores for each reference / hypothesis pairs in references and hypotheses
    Args:
       references: List of reference (assumed to be tokenized)
       hypothesis: List of hypotheses (assumed to be tokenized)
    '''

    if len(references) != len(hypotheses):
        raise RuntimeError(f'references {len(references)} and hypotheses {len(hypotheses)} are lists that should be of same size')

    scores = []

    for i, reference in enumerate(references):

        ref = reference.strip()
        if not ref:
            raise RuntimeError(f'reference is empty of valid words for reference at index {i}')

        ref_ = ref
        hyp_ = hypotheses[i]

        ter = sacrebleu.sentence_ter(hyp_, [ref_])  # hyp, [ref]: in this order and list '[ref]' is important!
        scores.append(ter.score)

    scores_np = np.array(scores)

    return scores_np


def webnlg_rdf_exactF1(references, hypotheses):
    ''' Computes exact F1 score for each reference / hypothesis pairs in references and hypotheses
    Args:
       references: List of reference (assumed to be tokenized)
       hypothesis: List of hypotheses (assumed to be tokenized)
    '''

    if len(references) != len(hypotheses):
        raise RuntimeError(f'references {len(references)} and hypotheses {len(hypotheses)} are lists that should be of same size')

    for i, ref in enumerate(references):
        hyp = hypotheses[i]
        assert isinstance(ref, list), 'references must be a list of list'
        assert isinstance(hyp, list), 'hypotheses must be a list of list'

    exactF1 = compute_exactF1(references, hypotheses)
    scores_np = np.array(exactF1)

    return scores_np
