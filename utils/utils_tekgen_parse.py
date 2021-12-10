import logging

import cfg

logger = logging.getLogger(__name__)
log = logger


def get_tekgen_triples(data):
    ''' Gets triples from generated data
    args:
      data: List of strings containing one or more triples
    returns
      triples_set: List of List of triples as a 'sub | rel | obj' string
    '''
    [subj_token, pred_token, obj_token] = cfg.SPECIAL_TOKENS_TEKGEN

    triples_set = []
    for sample in data:
        sample = sample.split(subj_token)
        if len(sample) != 2:
            triples_set.append([])
            continue

        rest = sample[1].strip()
        subject = rest.split(pred_token)[0]
        rest = rest.split(pred_token)[1:]

        triples = []
        for triple in rest:
            triple = triple.strip()
            if obj_token not in triple:
                continue

            triple = triple.split(obj_token)

            if len(triple) != 2:
                continue

            reln = triple[0].strip()
            obj = triple[1].strip()

            triple_str = f"{subject} | {reln} | {obj}"
            triples.append(triple_str)

        triples_set.append(triples)

    return triples_set
