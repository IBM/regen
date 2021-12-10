import logging

import torch
from torch.utils.data import Dataset
from dataset.webnlg.benchmark_reader import Benchmark, select_files
from dataset.prepare_dataset import get_prepare_class
from model.model import get_tokenizer_name
from dataset.webnlg_cards import get_webnlg_dataset_card
import cfg

logger = logging.getLogger(__name__)
log = logger

SPECIAL_TOKENS_WEBNLG = cfg.SPECIAL_TOKENS_WEBNLG


class WebNLGDataset(Dataset):
    ''' WebNLG Dataset'''

    def __init__(self, split, tokenizer, src, prepare='webnlg', prepare_args=None):
        '''
        Args:
          split : dataset split
          tokenizer : tokenizer to use
          src : source modality as defined in {root}/cfg.py
          prepare : prepare engine
          prepare_args : prepare engine arguments to propagate
        '''

        self.tokenizer = tokenizer
        self.split = split
        self.src = src

        self.raw = []  # raw data from dataset files
        self.samples = None  # prepared data samples List
        self.tokenizer_name = None

        self.meta_map = None  # map sample_idx -> meta information {'eid', 'category', 'size', 'shape', 'shape_type'}

        self.categories = None  # List of categories: List, sorted list of unique categories
        self.prepare = prepare  # Name of the preparation engine requested
        self.prepare_engine = None  # Preparation engine instance
        self.dataset_card = None  # Dataset info card

        # preparation engine
        self.prepare_engine = get_prepare_class(self.prepare, self.split, args=prepare_args)

        # webnlg dataset card
        self.dataset_card = get_webnlg_dataset_card(prepare)

        # tokenizer
        tokenizer_name = get_tokenizer_name(tokenizer)
        if tokenizer_name is None:
            raise ValueError(
                f"tokenizer name '{tokenizer_name}' is not recognized")
        self.tokenizer_name = tokenizer_name

        if self.src not in ['A', 'B', 'A+B']:
            raise RuntimeError(f" src/task '{self.src}' not valid")

        # prepare data
        self.load_data()
        prepare_dict = self.prepare_engine.prepare_data(self.raw)
        self.samples = prepare_dict['samples']
        self.categories = prepare_dict['categories']
        self.meta_map = prepare_dict['meta_map']

        self.update_tokenizer()

    def __getitem__(self, index):
        entry = self.samples[index]
        return entry

    def __len__(self):
        return len(self.samples)

    def load_data(self):
        ''' Loads data from input .xml files '''
        # grab all XML files
        fnames = self.get_files(self.split)

        benchmark = Benchmark()
        benchmark.fill_benchmark(fnames)

        for eid, entry in enumerate(benchmark.entries):
            texts, triples = [], []
            for _, lexic in enumerate(entry.lexs):
                text = lexic.lex
                texts.append(text)

            for _, triple in enumerate(entry.modifiedtripleset.triples):
                triple = (triple.s, triple.p, triple.o)
                triples.append(triple)

            category = entry.category
            eid = entry.id
            shape = entry.shape
            shape_type = entry.shape_type
            size = entry.size
            raw = {"text": texts,  # List of lexicalizations
                   "triple": triples,  # List of triples
                   "category": category,  # meta info for each entry
                   "eid": eid,
                   "shape": shape,
                   "shape_type": shape_type,
                   "size": size}

            self.raw.append(raw)

        return

    def update_tokenizer(self):
        ''' Updates tokenized by adding special tokens'''
        self.tokenizer.add_tokens(SPECIAL_TOKENS_WEBNLG)

    def get_category(self, idx):
        ''' Returns category for given sample index idx'''
        return self.meta_map[idx]['category']

    def get_meta(self, idx):
        ''' Returns meta information for given sample index idx'''
        return self.meta_map[idx]

    def get_categories(self):
        ''' Returns (sorted) list of categories for dataset'''
        return self.categories

    def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1):
        '''
        Inspired from : https://github.com/znculee/finetune-transformers/blob/master/dataset.py

        Overload the method of adding special tokens
        e.g.
        BART: <bos_id> token_ids_0 <eos_id>
        T5:   <pad_id> token_ids_0 <eos_id>
        '''

        if self.tokenizer_name == "t5":
            prefix_tokens = [self.tokenizer.pad_token_id]  # T5
        elif self.tokenizer_name == "bart":
            prefix_tokens = [self.tokenizer.bos_token_id]  # BART
        else:
            raise ValueError(f'tokenizer_name {self.tokenizer_name} is not recognized')

        suffix_tokens = [self.tokenizer.eos_token_id]

        if token_ids_1 is None:
            return prefix_tokens + token_ids_0 + suffix_tokens
        else:
            raise Exception("can't provide pair sequences")

    def _collate_fn(self, batch_samples):
        ''' Puts together minibatches '''

        # gather text and triples for batch

        texts = []
        triples = []
        samples_idx = []

        for sample in batch_samples:
            texts.append(sample['text'])
            triples.append(sample['triple'])
            samples_idx.append(sample['sample_idx'])

        # create source and target detepending on model and task

        source_samples = []
        target_samples = []

        if self.tokenizer_name == 't5':

            task_prefix_A = cfg.task_prefix_A
            task_prefix_B = cfg.task_prefix_B

            if self.src == 'A':  # A->B: text to graph

                # [Translate Text to Graph] A : B

                for txt, kb in zip(texts, triples):
                    source_samples.append(' '.join([task_prefix_A, txt]))
                    target_samples.append(kb)

            elif self.src == 'B':  # B->A: graph to text

                # [Translate Graph to Text] B : A

                for txt, kb in zip(texts, triples):
                    source_samples.append(' '.join([task_prefix_B, kb]))
                    target_samples.append(txt)

            elif self.src == 'A+B':  # A->B and B->A are part of minib

                # [Translate Text to Graph] A : B
                # [Translate Graph to Text] B : A

                # A->B
                for txt, kb in zip(texts, triples):
                    source_samples.append(' '.join([task_prefix_A, txt]))
                    target_samples.append(kb)

                # B->A
                for txt, kb in zip(texts, triples):
                    source_samples.append(' '.join([task_prefix_B, kb]))
                    target_samples.append(txt)
            else:

                raise RuntimeError(f'unknown task for src {self.src} for model {self.tokenizer_name}')

        elif self.tokenizer_name == 'bart':

            for sample in batch_samples:
                texts.append(sample["text"])
                triples.append(sample["triple"])

            if self.src == 'A':

                source_samples = texts[:]
                target_samples = triples[:]

            elif self.src == 'B':

                source_samples = triples[:]
                target_samples = texts[:]

            elif self.src == 'A+B':

                # A->B
                source_samples = texts[:]
                target_samples = triples[:]

                # B->A
                for txt, kb in zip(texts, triples):
                    source_samples.append(kb)
                    target_samples.append(txt)
            else:

                raise RuntimeError(f'unknown task for src {self.src} for model {self.tokenizer_name}')

        else:

            raise RuntimeError(f'unknown model {self.tokenizer_name}')

        self.tokenizer.build_inputs_with_special_tokens = self._build_inputs_with_special_tokens

        source_batch = self.tokenizer(
            source_samples,
            add_special_tokens=True,
            padding=True,
            return_tensors='pt',
            verbose=True
        )

        target_batch = self.tokenizer(
            target_samples,
            add_special_tokens=True,
            padding=True,
            return_tensors='pt'
        )

        samples_idx_tensor = torch.LongTensor(samples_idx)
        batch = (source_batch['input_ids'],
                 source_batch['attention_mask'],
                 target_batch['input_ids'],
                 target_batch['attention_mask'],
                 samples_idx_tensor)

        return batch

    def get_files(self, split):
        ''' Get files for given split '''

        path_to_corpus, file_name = None, None

        if split == "train":
            path_to_corpus = self.dataset_card.train_dir
        elif split == "val":
            path_to_corpus = self.dataset_card.val_dir
        elif split == 'testA':  # A->B t2g
            path_to_corpus = self.dataset_card.test_dir
            file_name = self.dataset_card.text2rdf_with_refs_xml
        elif split == 'testB':  # B->A g2t
            path_to_corpus = self.dataset_card.test_dir
            file_name = self.dataset_card.rdf2text_with_refs_xml
        else:
            raise RuntimeError(f'invalid split: [{split}]')

        if file_name is None:
            # case for train/val
            files = select_files(path_to_corpus)
        else:
            # case for test
            files = [(path_to_corpus, file_name)]

        return files
