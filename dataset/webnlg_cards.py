import logging
import os
import cfg

logger = logging.getLogger(__name__)
log = logger


def get_webnlg_dataset_card(prepare_engine):
    ''' Gets a dataset card for webnlg depending on prepare engine '''

    if prepare_engine not in cfg.PREPARE_ENGINES:
        raise ValueError(
            f"prepare_engine '{prepare_engine}' is not valid. shoud be in {cfg.PREPARE_ENGINES}")

    dataset_card = None

    if prepare_engine == 'webnlg':
        dataset_card = WebNLGCard
    elif prepare_engine == 'webnlg-lex1':
        dataset_card = WebNLGLex1Card
    elif prepare_engine == 'webnlg-lex1-kbpm':
        dataset_card = WebNLGLex1KBPermCard
    else:
        raise ValueError(
            f"prepare_engine '{prepare_engine}' is not valid")

    return dataset_card


class WebNLGCard(object):
    ''' WebNLG dataset description card '''

    base_dir = cfg.datasets_webnlg_path  # '{root_dir}/corpora/webnlg-dataset'
    meta_dir = cfg.datasets_webnlg_meta_path  # '{root_dir}/corpora/webnlg-references'
    release = 'release_v3.0'

    corpus_dir = os.path.join(base_dir, release)  # '{root_dir}/corpora/webnlg-dataset/release_v3.0'
    corpus_meta_dir = os.path.join(meta_dir, release)  # '{root_dir}/corpora/webnlg-references/release_v3.0'

    train_dir = os.path.join(corpus_dir, 'en/train/')
    val_dir = os.path.join(corpus_dir, 'en/dev/')
    test_dir = os.path.join(corpus_dir, 'en/test/')

    text2rdf_with_refs_xml = 'semantic-parsing-test-data-with-refs-en.xml'  # t2g
    rdf2text_with_refs_xml = 'rdf-to-text-generation-test-data-with-refs-en.xml'  # g2t

    # G2T: RDF2Text (text) references
    references_val = os.path.join(corpus_meta_dir, 'en/references.val')  # reference dir (text)
    references_test = os.path.join(corpus_meta_dir, 'en/references.test')  # reference dir (text)

    # T2G: Text2RDF rdf (graph) references
    # Note: references_rdf_test is derived from text2rdf_with_refs_xml, the content is semantically the same (but *not* byte identical)
    references_rdf_val_xml = os.path.join(corpus_meta_dir, 'en/references.rdf.val/val.xml')  # reference xml (text2rdf)
    # references_rdf_test = os.path.join(corpus_meta_dir, 'en/references.rdf.test/test.xml')  # reference xml (rdf)
    references_rdf_test_xml = os.path.join(corpus_dir, 'en/test/semantic-parsing-test-data-with-refs-en.xml')  # reference xml (all)
    references_rdf_test_seen_xml = os.path.join(corpus_meta_dir, 'en/references.rdf.test.seen_unseen/seen.semantic-parsing-test-data-with-refs-en.xml')  # reference xml (seen)
    references_rdf_test_unseen_xml = os.path.join(corpus_meta_dir, 'en/references.rdf.test.seen_unseen/unseen.semantic-parsing-test-data-with-refs-en.xml')  # reference xml (unseen)

    val_triple_set = 1667
    test_triple_set = 1779
    train_triple_set = 13211
    testA_set = 2155  # t2g text2rdf (semantic parsing) -> 2155 individual lexicalization (text) per RDF
    testB_set = 1779  # g2t rdf2text -> 1779 RDFs w/ one or more lexicalizations each.


class WebNLGLex1Card(WebNLGCard):
    ''' WebNLG dataset description card for WebNLG w/ 1 sentence per sample'''

    references_val = None
    references_test = None
    val_triple_set = 4464  # original 1667
    test_triple_set = 5150  # original 1779
    train_triple_set = 35426  # original 13211
    testA_set = 2155  # original 2155
    testB_set = 5150  # original 1779


class WebNLGLex1KBPermCard(WebNLGCard):
    ''' WebNLG dataset description card for WebNLG w/ 1 lexicalization per sample'''

    references_val = None
    references_test = None
    val_triple_set = 8165  # original 1667
    test_triple_set = 10168  # original 1779
    train_triple_set = 64739  # original 13211
    testA_set = 4263  # original 2155
    testB_set = 10168  # original 1779
