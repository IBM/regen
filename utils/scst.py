""" SCST-related utils
"""

import logging
logger = logging.getLogger(__name__)
log = logger


def log_scst(greedy_seq, target_seq, sample_seq, rewards, msg=None):
    '''
    Logging for SCST given sequences and rewards
    '''

    msg = f'{msg} ' if msg is not None else ''

    reward_score = rewards['reward'].tolist()
    reward_names = list(rewards['rewards'].keys())

    for i, (g, t, s, r) in enumerate(zip(greedy_seq, target_seq, sample_seq, reward_score)):

        log.info(f'{msg}----')
        log.info(f'{msg}b{i}: greedy: {g}')
        log.info(f'{msg}b{i}: sample: {s}')
        log.info(f'{msg}b{i}: target: {t}')
        log.info(f'{msg}b{i}: --')
        log.info(f'{msg}b{i}: reward: {r}')
        for rn in reward_names:
            values = rewards['rewards'][rn]
            log.info(f'{msg}b{i}: {rn}: {values[i]}')
        log.info(f'{msg}----')
