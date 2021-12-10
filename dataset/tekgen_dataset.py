import logging

import json
import torch
from torch.utils.data import Dataset

from dataset.prepare_dataset import get_prepare_class
from model.model import get_tokenizer_name
from dataset.tekgen_cards import get_tekgen_dataset_card
import cfg

logger = logging.getLogger(__name__)
log = logger

SPECIAL_TOKENS_TEKGEN = cfg.SPECIAL_TOKENS_TEKGEN


class TekGenDataset(Dataset):
    ''' TekGen Dataset'''

    def __init__(self, split, tokenizer, src, prepare='tekgen_default', prepare_args=None, max_seq_len=256):
        '''
        Args:
          split : dataset split
          tokenizer : tokenizer to use
          src : source modality as defined in {root}/cfg.py
          prepare : prepare engine
          prepare_args : prepare engine arguments to propagate
          max_seq_len : tokenizer max length
        '''

        self.tokenizer = tokenizer
        self.split = split
        self.src = src
        self.max_seq_len = max_seq_len

        self.raw = []  # raw data from dataset files
        self.samples = None  # prepared data samples List
        self.tokenizer_name = None

        self.meta_map = None
        self.prepare = prepare  # name of the preparation engine requested
        self.prepare_engine = None  # preparation engine intance
        self.dataset_card = None

        # preparation engine
        self.prepare_engine = get_prepare_class(self.prepare, self.split, args=prepare_args)

        # tekgen dataset card
        self.dataset_card = get_tekgen_dataset_card(prepare)

        # tokenizer
        tokenizer_name = get_tokenizer_name(tokenizer)
        if tokenizer_name is None:
            raise ValueError(
                f"tokenizer name '{tokenizer_name}' is not recognized")
        self.tokenizer_name = tokenizer_name

        if self.src not in ['A', 'B', 'A+B']:
            raise RuntimeError(f" src/task '{self.src}' not valid")

        # prepare data
        self.load_data(split)
        prepare_dict = self.prepare_engine.prepare_data(self.raw)
        self.samples = prepare_dict['samples']
        self.meta_map = prepare_dict['meta_map']

        self.update_tokenizer()

    def __getitem__(self, index):
        entry = self.samples[index]
        return entry

    def __len__(self):
        return len(self.samples)

    def load_data(self, split):
        ''' Loads data for a given split '''

        if split == "train":
            fname = self.dataset_card.train_file
        elif split == 'val':
            fname = self.dataset_card.valid_file
        elif split in ['testA', 'testB']:
            fname = self.dataset_card.test_file
        else:
            raise Exception(f"{split} not defined in {self.dataset_card}")

        if self.prepare == 'tekgen-default':
            return self.load_data_modified(fname)

        elif self.prepare == 'tekgen-official':
            return self.load_data_official(fname)

        else:
            raise Exception(f'dataset category incorrect: [{self.dataset_category}]')

    def load_data_official(self, fname):
        ''' Loads data for "official" TekGen release '''

        with open(fname) as file:
            for entry in file:
                entry = entry.strip()
                entry = json.loads(entry)
                raw = {'sentence': entry['sentence'],
                       'triple_text': entry['serialized_triples'],
                       'subject': entry['triples'][0][0],
                       'entity_object_pairs': []}

                for reln_obj in entry['triples']:
                    reln_obj = {'relation': reln_obj[1], 'object': reln_obj[2]}
                    raw['entity_object_pairs'].append(reln_obj)
                self.raw.append(raw)

    def load_data_modified(self, fname):
        ''' Lods data from file w/ modified representation '''

        with open(fname, 'r', encoding="utf-8") as fin:
            data = json.load(fin)

        for entry in data:
            raw = {'sentence': entry['sentence'],
                   'triple_text': entry['triple'],
                   'subject': entry['subject'],
                   'entity_object_pairs': entry['reln_obj']}
            self.raw.append(raw)

        return

    def update_tokenizer(self):
        '''Update tokenizer by adding special tokens'''
        self.tokenizer.add_tokens(SPECIAL_TOKENS_TEKGEN)

    def get_meta(self, idx):
        '''Returns meta information for given sample index idx'''
        return self.meta_map[idx]

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
        '''Puts together minibatch'''

        # gather text and triples for batch
        texts = []
        triples = []
        samples_idx = []

        for sample in batch_samples:
            texts.append(sample['text'])
            triples.append(sample['triple'])
            samples_idx.append(sample['sample_idx'])

        # create source and target depending on model and task
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
            padding='max_length',
            return_tensors='pt',
            verbose=True,
            max_length=self.max_seq_len,
            truncation=True,
        )

        target_batch = self.tokenizer(
            target_samples,
            add_special_tokens=True,
            padding=True,
            return_tensors='pt',
            max_length=self.max_seq_len,
            truncation=True,
        )

        samples_idx_tensor = torch.LongTensor(samples_idx)
        batch = (source_batch['input_ids'],
                 source_batch['attention_mask'],
                 target_batch['input_ids'],
                 target_batch['attention_mask'],
                 samples_idx_tensor)

        return batch
