#!/usr/bin/env python
# coding=utf-8

from benchmark_reader import Benchmark
from benchmark_reader import select_files


def create_reference_files(b, lang):
    """
    Write reference files for evaluation.
    :param b: instance of Benchmark class
    :param lang: en, ru
    """
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
        with open(f'reference{str(j)}-{lang}.txt', 'w+') as f:
            out = []
            for ref in target_out:
                try:
                    out.append(ref[j] + '\n')
                except IndexError:
                    out.append('\n')
            f.write(''.join(out))


def run_on_corpus_per_lang(path_to_corpus, lang):
    # initialise Benchmark object
    b = Benchmark()
    # collect xml files
    files = select_files(path_to_corpus)
    # load files to Benchmark
    b.fill_benchmark(files)
    # generate references
    create_reference_files(b, lang)


# where to find the corpus
# English
path = './challenge2020_train_dev_v2/en/dev'
run_on_corpus_per_lang(path, 'en')
# Russian
path = './challenge2020_train_dev_v2/ru/dev'
run_on_corpus_per_lang(path, 'ru')
