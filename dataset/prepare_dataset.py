import logging
from itertools import permutations
import random
import cfg

logger = logging.getLogger(__name__)
log = logger


def get_prepare_class(name, split, args):
    ''' Returns the preparation class given the name of the preparation engine '''

    if name not in cfg.PREPARE_ENGINES:
        raise ValueError(
            f"preparation engine '{name}' is not valid. shoud be in {cfg.PREPARE_ENGINES}")

    engine_class = None

    if name == 'webnlg':
        engine_class = PrepareWebNLG(split)
    elif name == 'webnlg-lex1':
        engine_class = PrepareWebNLGLex1(split)
    elif name == 'webnlg-lex1-kbpm':
        engine_class = PrepareWebNLGLex1KBPerm(split)
    elif name == 'tekgen-default':
        engine_class = PrepareTekGen(split)
    elif name == 'tekgen-official':
        engine_class = PrepareTekGen(split)
    else:
        raise ValueError(
            f"preparation engine '{name}' for split '{split}' is not valid")

    if engine_class is None:
        raise RuntimeError(
            f"preparation engine '{name}' for split '{split}' did not generate a valid class")

    return engine_class


class Prepare():
    ''' Prepare base class definition:
    This class exists only to show the expected API.  It is a no-op in
    terms of preparation of the data otherwise.  The class stores the
    value of the dataset split.

    Description:
    The data is basically untouched. Preparation is only the joining
    of triples and lexicalizations into one single string, one for all
    the triples, and one for all lexicalizations.
    '''

    def __init__(self, split):
        self.split = split

    def prepare_data(self, raw_data):
        ''' Prepares raw data into final samples
        Args:
        - raw_data: list of raw samples
        Returns:
        - dict: dict['samples']: prepared samples where
        dict['samples'] is a list of samples as dict: {'text':
        text_sample, 'triple': triple_sample}
        '''
        samples = []
        for idx, raw in enumerate(raw_data):
            texts = raw['text']
            triples = raw['triple']
            texts_sample = self.prepare_texts(texts)
            triples_sample = self.prepare_triples(triples)

            sample = {'text': texts_sample, 'triple': triples_sample, 'sample_idx': idx}
            sample = self.process_sample(sample)

            samples.append(sample)

        return {'samples': samples}

    def prepare_texts(self, texts):
        ''' Text preparation. Given a list of text, return a list of formated text'''

        texts_samples = []

        for text in texts:
            text_fmt = self.format_text(text)
            texts_samples.append(text_fmt)

        return texts_samples

    def prepare_triples(self, triples):
        ''' Triples preparation. Given a list of triples, returns a list of formated triples. '''

        triples_samples = []

        for triple in triples:
            triple_fmt = self.format_triple(triple)
            triples_samples.append(triple_fmt)

        return triples_samples

    def format_text(text):
        ''' Formats text '''
        return text

    def format_triple(triple):
        ''' Formats a triple'''
        return triple

    def process_sample(self, sample):
        ''' Processes list of texts and list of triples of a sample'''

        sample['text'] = ' '.join(sample['text'])
        sample['triple'] = ' '.join(sample['triple'])

        return sample


class PrepareTekGen():
    ''' Prepare Class for TekGen '''

    def __init__(self, split):
        self.split = split
        self.SPECIAL_TOKENS = cfg.SPECIAL_TOKENS_TEKGEN

    def prepare_data(self, raw_data):
        ''' Prepares raw data into final samples
        Args:
        - raw_data: list of raw samples
        Returns:
        - dict: dict['samples']: prepared samples where
        dict['samples'] is a list of samples as dict: {'text':
        text_sample, 'triple': triple_sample}
        '''
        samples = []
        meta_map = {}
        for idx, raw in enumerate(raw_data):
            sentence = raw['sentence']
            triples = raw['entity_object_pairs']
            subject = raw['subject']

            sentence = sentence
            triples = [subject, triples]
            sentence_sample = self.prepare_text(sentence)
            triples_sample = self.prepare_triple(triples)

            sample = {'text': sentence_sample, 'triple': triples_sample, 'sample_idx': idx}

            samples.append(sample)
            meta_map[idx] = {'sentence': raw['sentence'], 'triple_text': raw['triple_text']}

        prepare_dict = {'samples': samples, "meta_map": meta_map}

        return prepare_dict

    def prepare_text(self, text):
        ''' Prepares text sample '''
        return text

    def prepare_triple(self, triple):
        ''' Prepares triplle sample '''
        subject, entity_object_pairs = triple
        formatted_triple = [self.SPECIAL_TOKENS[0], subject]

        for idx in range(len(entity_object_pairs)):
            formatted_triple.append(self.SPECIAL_TOKENS[1])
            formatted_triple.append(entity_object_pairs[idx]['relation'])
            formatted_triple.append(self.SPECIAL_TOKENS[2])
            formatted_triple.append(entity_object_pairs[idx]['object'])

        return ' '.join(formatted_triple)


class PrepareWebNLG(Prepare):
    '''Prepare for WebNLG '''

    def __init__(self, split):
        super().__init__(split)
        self.SPECIAL_TOKENS = cfg.SPECIAL_TOKENS_WEBNLG

    def prepare_data(self, raw_data):
        ''' Prepares raw data into final samples
        Args:
        - raw_data: list of raw samples + meta info
        Returns:
        - samples: prepared samples
        - meta_map: dict that maps index of sample (from raw) to meta info dict
        - categories: sorted list of categories

        Description:
        The data for text is untouched. Triples are then expanded as
        <s> subject <p> predicate <o> object.
        Triples and lexicalizations are joined into single strings, one for
        all the triples, and one for all lexicalizations.
        Example:
        (s1,p1,o1) (s2,p2,o2), "lex1" "lex2" "lex3" ->
        "<s> s1 <p> p1 <o> o1 <s> s2 <p> p2 <o> o2", "lex1 lex2 lex3"

        '''

        samples = []
        meta_map = {}
        cat_set = set()

        for idx, raw in enumerate(raw_data):
            texts = raw['text']
            triples = raw['triple']

            # meta information
            category = raw['category']
            eid = raw['eid']
            shape = raw['shape']
            shape_type = raw['shape_type']
            size = raw['size']

            # dict of meta information for sample/entry
            meta = {'category': category, 'eid': eid, 'shape': shape, 'shape_type': shape_type, 'size': size}

            cat_set.add(category)

            meta_map[idx] = meta
            texts_sample = self.prepare_texts(texts)
            triples_sample = self.prepare_triples(triples)

            sample = {'text': texts_sample, 'triple': triples_sample, 'sample_idx': idx}
            sample = self.process_sample(sample)

            samples.append(sample)

        categories = sorted(list(cat_set))

        prepare_dict = {'samples': samples,
                        'categories': categories,
                        'meta_map': meta_map}

        return prepare_dict

    def prepare_texts(self, texts):
        ''' Texts preparation. Given a list of text, return a list of formated text'''

        texts_samples = []

        for text in texts:
            text_fmt = self.format_text(text)
            texts_samples.append(text_fmt)

        return texts_samples

    def prepare_triples(self, triples):
        ''' Triples preparation. Given a list of triples, returns a list of formated triples. '''

        triples_samples = []

        for triple in triples:
            triple_fmt = self.format_triple(triple)
            triples_samples.append(triple_fmt)

        return triples_samples

    def format_text(self, text):
        return text

    def format_triple(self, triple):
        ''' Formats input triple s,o,p as
            "<s> s <p> p <o> o"
        where
            s, p, o are subject, predicate, object
            <s>, <p>, <o> are tags defined from self.SPECIAL_TOKENS
        '''
        if len(triple) != len(self.SPECIAL_TOKENS):
            raise RuntimeError(
                f"triple length [{len(triple)}] and special tokens do not match length [{len(self.SPECIAL_TOKENS)}] for {self.SPECIAL_TOKENS}")

        formatted_triple = []
        for idx in range(len(triple)):
            formatted_triple.append(self.SPECIAL_TOKENS[idx])
            formatted_triple.append(triple[idx])

        return ' '.join(formatted_triple)

    def process_sample(self, sample):
        '''Processes texts and triples of a sample'''

        sample['text'] = ' '.join(sample['text'])
        sample['triple'] = ' '.join(sample['triple'])

        return sample


class PrepareWebNLGLex1(PrepareWebNLG):
    '''Prepare for WebNLG using 1 lexicalization per sample
    '''

    def __init__(self, split):
        super().__init__(split)
        pass

    def prepare_data(self, raw_data):
        ''' Prepares raw data into final samples
        Args:
        - raw_data: list of raw samples
        Returns:
        - samples: prepared samples
        - meta_map: dict that maps index of sample (from raw) to meta info dict
        - categories: sorted list of categories

        Description:
        Goal is to allow for using original test set for WebNLG 2020 Challenge comparison
        1. one sample per lexicalization

        Example:
        (s1,p1,o1) (s2,p2,o2), "lex1" "lex2" "lex3" ->
        "<s> s1 <p> p1 <o> o1 <s> s2 <p> p2 <o> o2", "lex1"
        "<s> s1 <p> p1 <o> o1 <s> s2 <p> p2 <o> o2", "lex2"
        "<s> s1 <p> p1 <o> o1 <s> s2 <p> p2 <o> o2", "lex3"
        '''

        samples = []
        meta_map = {}
        cat_set = set()

        for _, raw in enumerate(raw_data):

            texts = raw['text']
            triples = raw['triple']

            # meta information
            category = raw['category']
            eid = raw['eid']
            shape = raw['shape']
            shape_type = raw['shape_type']
            size = raw['size']

            # dict of meta information for sample/entry
            meta = {'category': category, 'eid': eid, 'shape': shape, 'shape_type': shape_type, 'size': size}

            cat_set.add(category)

            texts_sample = self.prepare_texts(texts)  # list of text
            triples_sample = self.prepare_triples(triples)  # list of triples

            for text in texts_sample:
                idx = len(samples)
                meta_map[idx] = meta

                sample = {'text': [text], 'triple': triples_sample, 'sample_idx': idx}
                sample = self.process_sample(sample)
                samples.append(sample)

        categories = sorted(list(cat_set))

        prepare_dict = {'samples': samples,
                        'categories': categories,
                        'meta_map': meta_map}

        return prepare_dict


class PrepareWebNLGLex1KBPerm(PrepareWebNLG):
    ''' Prepare for WebNLG using 1 lexicalization per sample, Triple permutations
    '''

    def __init__(self, split):
        super().__init__(split)
        pass

    def prepare_data(self, raw_data):
        ''' Prepares raw data into final samples
        Args:
        - raw_data: list of raw samples
        Returns:
        - samples: prepared samples
        - meta_map: dict that maps index of sample (from raw) to meta info dict
        - categories: sorted list of categories

        Description:
        Goal is to allow for using original test set for WebNLG comparision
        1. one sample per lexicalization
        2. permutations of triples

        Example:
        (s1,p1,o1) (s2,p2,o2), "lex1" "lex2" "lex3" ->
        "<s> s1 <p> p1 <o> o1 <s> s2 <p> p2 <o> o2", "lex1"
        "<s> s2 <p> p2 <o> o2 <s> s1 <p> p1 <o> o1", "lex1"
        "<s> s1 <p> p1 <o> o1 <s> s2 <p> p2 <o> o2", "lex2"
        "<s> s2 <p> p2 <o> o2 <s> s1 <p> p1 <o> o1", "lex2"
        "<s> s1 <p> p1 <o> o1 <s> s2 <p> p2 <o> o2", "lex3"
        "<s> s2 <p> p2 <o> o2 <s> s1 <p> p1 <o> o1", "lex3"
        '''

        samples = []
        meta_map = {}
        cat_set = set()

        # The magic numbers decide how many permutations we sample
        # reminder: 7!= 5040 6!=720 5!=120 4!=24 3!=6 2!=2
        max_permutations = {
            '1': 1,
            '2': 2,
            '3': 2,
            '4': 2,
            '5': 2,
            '6': 4,
            '7': 3}

        for _, raw in enumerate(raw_data):

            texts = raw['text']
            triples = raw['triple']

            # meta information
            category = raw['category']
            eid = raw['eid']
            shape = raw['shape']
            shape_type = raw['shape_type']
            size = raw['size']

            # dict of meta information for sample/entry
            meta = {'category': category, 'eid': eid, 'shape': shape, 'shape_type': shape_type, 'size': size}

            cat_set.add(category)

            texts_sample = self.prepare_texts(texts)  # list of text
            triples_sample = self.prepare_triples(triples)  # list of triples

            # 1 lex per sample
            for text in texts_sample:

                # triples permutations
                kb_perm = list(permutations(triples_sample))  # expand all permuted triples
                if self.split == 'train':
                    random.shuffle(
                        kb_perm)  # in-place shuffling for train only, for val, test, we want the *same* order

                # we only keep the first max_perm permuted triples
                max_perm = max_permutations[size]

                for triples_perm in kb_perm[:max_perm]:
                    idx = len(samples)
                    meta_map[idx] = meta

                    sample = {'text': [text], 'triple': triples_perm, 'sample_idx': idx}
                    sample = self.process_sample(sample)
                    samples.append(sample)

        categories = sorted(list(cat_set))

        prepare_dict = {'samples': samples,
                        'categories': categories,
                        'meta_map': meta_map}

        return prepare_dict
