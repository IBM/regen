from typing import List, Any
import os

from addict import Dict as adict
from xml.etree.ElementTree import Element, SubElement

from utils.utils import save_json
from utils.utils_xml import xml_prettify
from utils.utils_webnlg import get_webnlg_rdf_generated_filenames
from utils.utils_tekgen_parse import get_tekgen_triples

import logging

logger = logging.getLogger(__name__)
log = logger


def create_tekgen_xml(data, meta_infos, ts_header, t_header):
    ''' Creates a benchmark from data'''
    benchmark = Element('benchmark')
    entries = SubElement(benchmark, 'entries')

    assert len(meta_infos) == len(data)

    for idx, triples in enumerate(data):

        entry = SubElement(entries, 'entry', attrib={'eid': str(idx)})
        t_entry = SubElement(entry, ts_header)

        for triple in triples:
            element = SubElement(t_entry, t_header)
            element.text = triple

    return benchmark


def save_tekgen_rdf(hyps: Any,
                    targets: Any,
                    meta_infos: List,
                    out_dir: str,
                    prepare: str = 'tekgen-default'):
    ''' Saves generated graph output for tekgen to .xml format '''

    tgts = get_tekgen_triples(targets)
    hyps = get_tekgen_triples(hyps)

    filesD = get_webnlg_rdf_generated_filenames(out_dir)
    tgts_triples_json = filesD.targets_triples_json
    hyps_triples_json = filesD.hypotheses_triples_json
    tgts_fname_xml = filesD.targets_xml
    hyps_fname_xml = filesD.hypotheses_xml

    save_json(tgts_triples_json, tgts, msg='parsed target triples:', logger=log)
    save_json(hyps_triples_json, hyps, msg='parsed hypothesis triples:', logger=log)

    if len(tgts) != len(hyps):
        raise Exception(f"targets size {len(tgts)} is not same as hypothesis size {len(hyps)}")

    tgts_xml_tree = create_tekgen_xml(tgts, meta_infos, "modifiedtripleset", "mtriple")
    hyps_xml_tree = create_tekgen_xml(hyps, meta_infos, "generatedtripleset", "gtriple")  # required by WebNLG 2020 Challenge

    log.info(f"# Creating targets xml  file : [{tgts_fname_xml}]")
    log.info(f"# Creating hypothesis xml file : [{hyps_fname_xml}]")

    with open(tgts_fname_xml, 'w', encoding='utf-8') as f:
        f.write(xml_prettify(tgts_xml_tree))

    with open(hyps_fname_xml, 'w', encoding='utf-8') as f:
        f.write(xml_prettify(hyps_xml_tree))

    return filesD


def tekgen_text_args(dir_name, split, subset='all'):
    ''' Returns a dot dict w/ parameters need for tekgen text evaluation (using webnlg tools)'''

    args = adict()
    args.language = 'en'

    args.reference = os.path.join(dir_name, 'target')
    args.hypothesis = os.path.join(dir_name, 'hypothesis')

    args.num_refs = 1
    args.metrics = 'bleu,meteor,chrf++'
    args.ncorder = 6
    args.nworder = 2
    args.beta = 2.0

    return args
