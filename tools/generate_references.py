'''Generate Reference for RDF2Text (G2T)

   Adapted from code in https://gitlab.com/webnlg/corpus-reader ,
   specifically: https://gitlab.com/webnlg/corpus-reader/-/blob/master/generate_references.py

   Generating references for the RDF-to-text evaluation
   See generate_references.py in corpus-reader link above.

Original code comes w/ MIT License below:

MIT License

Copyright (c) 2020 WebNLG

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


'''
import logging
import logging.handlers

import argparse
import os
from utils.utils import mkdir_p

from dataset.webnlg.benchmark_reader import Benchmark
from dataset.webnlg.benchmark_reader import select_files
from dataset.webnlg_cards import WebNLGCard

# ------------------------------
# Logger
# ------------------------------
# logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.propagate = False  # do not propagate logs to previously defined root logger (if any).
formatter = logging.Formatter('%(asctime)s - %(levelname)s(%(name)s): %(message)s')
# console
consH = logging.StreamHandler()
consH.setFormatter(formatter)
consH.setLevel(logging.INFO)
logger.addHandler(consH)
# file handler
request_file_handler = True
log = logger


def create_reference_files(b, lang, out_dir):
    '''
    Write reference files for evaluation.
    :param b: instance of Benchmark class
    :param lang: en, ru
    note: from WebNLG code repos
    '''
    target_out = []  # List[List]
    for iterator, entry in enumerate(b.entries):
        entry_refs = []
        if lang == 'en':
            for lex in entry.lexs:
                entry_refs.append(lex.lex)
        elif lang == 'ru':
            for lex in entry.lexs[1::2]:  # take every second lexicalisation, i.e. only ru
                entry_refs.append(lex.lex)
        else:
            print('unknown language')
        # take only unique references (needed for Russian)
        unique_refs = list(dict.fromkeys(entry_refs))
        target_out.append(unique_refs)
    # the list with max elements
    max_refs = sorted(target_out, key=len)[-1]
    # write references files
    for j in range(0, len(max_refs)):
        # fname = f'reference{str(j)}-{lang}.txt'
        fname = f'reference{str(j)}'
        fpath = os.path.join(out_dir, fname)
        with open(fpath, 'w+') as f:
            out = []
            for ref in target_out:
                try:
                    out.append(ref[j] + '\n')
                except IndexError:
                    out.append('\n')
            f.write(''.join(out))
        log.info(f'{fpath} [{len(out)} lines]')


def run_on_corpus_per_lang(path_to_corpus, lang, split, out_dir):
    # initialise Benchmark object
    b = Benchmark()

    # collect xml files
    if split == 'val' or split == 'train':
        files = select_files(path_to_corpus)
    elif split == 'test':
        files = [(path_to_corpus, WebNLGCard.rdf2text_with_refs_xml)]
    else:
        raise RuntimeError(f'invalid split: [{split}]')

    # load files to Benchmark
    b.fill_benchmark(files)
    # generate references
    create_reference_files(b, lang, out_dir)


def main(args):
    ''' Create reference files for evaluation WebNLG tools'''

    if request_file_handler:
        # file handler
        base = os.path.basename(__file__)
        scp_dir = os.path.realpath(__file__)
        log_dir = os.path.realpath(os.path.join(scp_dir, '..', 'logs'))
        mkdir_p(log_dir, logger=log)
        log_filename = os.path.join(log_dir, f'{base}.log')
        fileH = logging.handlers.RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=1)
        fileH.setFormatter(formatter)
        fileH.setLevel(logging.DEBUG)
        # add handler
        logger.addHandler(fileH)
        log.info('# File logger started')

    path_to_corpus = None

    if args.split == "train":
        path_to_corpus = WebNLGCard.train_dir
    elif args.split == "val":
        path_to_corpus = WebNLGCard.val_dir
    elif args.split == 'test':
        path_to_corpus = WebNLGCard.test_dir
    else:
        raise RuntimeError(f'invalid split: [{args.split}]')

    mkdir_p(args.output_dir)
    run_on_corpus_per_lang(path_to_corpus, 'en', args.split, args.output_dir)

    return


if __name__ == "__main__":

    # Parsing arguments
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--split", type=str, choices=['val', 'test'], default='', help="split to create references for 'val'|test'")
    parser.add_argument("--output_dir", type=str, default='./generate_references', help="where to output the reference files")
    args = parser.parse_args()

    main(args)
