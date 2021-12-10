import os
import logging
import cfg

logger = logging.getLogger(__name__)
log = logger


def get_tekgen_dataset_card(prepare_engine):
    ''' Gets a tekgen dataset card given a preparation engine '''

    if prepare_engine not in cfg.PREPARE_ENGINES:
        raise ValueError(
            f"prepare_engine '{prepare_engine}' is not valid. should be in {cfg.PREPARE_ENGINES}")

    dataset_card = None

    if prepare_engine == 'tekgen-default':
        dataset_card = TekGenDefaultCard
    elif prepare_engine == 'tekgen-official':
        dataset_card = TekGenOfficialCard
    else:
        raise ValueError(
            f"prepare_engine '{prepare_engine}' is not valid")

    return dataset_card


class TekGenDefaultCard(object):
    ''' TekGen dataset description card '''

    base_dir = cfg.datasets_tekgen_default_path
    train_file = os.path.join(base_dir, 'tekgen_train.json')
    valid_file = os.path.join(base_dir, 'tekgen_validation.json')
    test_file = os.path.join(base_dir, 'tekgen_test.json')

    train_triple_set = 6368330
    valid_triple_set = 796116
    test_triple_set = 796159

    testA_set = test_triple_set
    testB_set = test_triple_set


class TekGenOfficialCard(object):
    ''' TekGen ("official") dataset description card '''

    base_dir = cfg.datasets_tekgen_official_path
    train_file = os.path.join(base_dir, 'quadruples-train.tsv')
    valid_file = os.path.join(base_dir, 'quadruples-validation.tsv')
    test_file = os.path.join(base_dir, 'quadruples-test.tsv')

    train_triple_set = 6368330
    valid_triple_set = 796116
    test_triple_set = 796159

    testA_set = test_triple_set
    testB_set = test_triple_set
