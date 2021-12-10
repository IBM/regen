import logging
import cfg

logger = logging.getLogger(__name__)
log = logger


def get_triples(data):
    ''' Returns triples from generated data
    args:
      data: List of strings containing one or more triples
    returns
      triples_set: List of List of triples as a 'sub | rel | obj' string
    '''
    [subj_token, pred_token, obj_token] = cfg.SPECIAL_TOKENS_WEBNLG

    triples_set = []
    for sample in data:
        triples = []
        sample = sample.split(subj_token)
        for triple in sample:
            triple = triple.strip()

            if len(triple) == 0:
                continue

            '''
                ill-formed if :
                    a) __predicate__ not present or present more than once
                    b) __object__ not present or present more than once
            '''
            if triple.count(pred_token) != 1:
                continue

            if triple.count(obj_token) != 1:
                continue

            triple = triple.replace(pred_token, "|")
            triple = triple.replace(obj_token, "|")
            triple = triple.strip()

            # check if it is valid triple
            if len(triple.split(" | ")) != 3:
                continue

            # check if entity/relation is valid
            invalid_ele = False
            for ele in triple.split(" | "):
                if len(ele.strip()) == 0:
                    invalid_ele = True

            if invalid_ele:
                continue

            triples.append(triple)

        triples_set.append(triples)

    return triples_set
