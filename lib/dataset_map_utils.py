'''
Methods for mapping mention IDs to dataset names, and dataset names to
use within evaluation framework
'''

import codecs

SemCor = 'SemCor'
SemEval2007 = 'SemEval2007'
SemEval2013 = 'SemEval2013'
SemEval2015 = 'SemEval2015'
SensEval2 = 'SensEval2'
SensEval3 = 'SensEval3'

TRAIN = set([
    SemCor
])
TEST = set([
    SemEval2007,
    SemEval2013,
    SemEval2015,
    SensEval2,
    SensEval3
])

DatasetOutputMap = {
    SemEval2007 : 'semeval2007',
    SemEval2013 : 'semeval2013',
    SemEval2015 : 'semeval2015',
    SensEval2 : 'senseval2',
    SensEval3 : 'senseval3'
}

def readDatasetMap(f, get_IDs=False, get_lemmas=False):
    _map = {}
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            (mention_ID, ds, instance_ID, lemma) = [s.strip() for s in line.split('\t')]
            stored = [ds]
            if get_IDs: stored.append(instance_ID)
            if get_lemmas: stored.append(lemma)
            if len(stored) == 1: stored = stored[0]
            _map[int(mention_ID)] = stored
    return _map

def assignTrainTest(_map):
    train, test = [], []
    for (mention_ID, ds) in _map.items():
        if ds in TRAIN:
            train.append(mention_ID)
        elif ds in TEST:
            test.append(mention_ID)
        else:
            raise KeyError('Unknown dataset "%s"' % ds)
    return train, test
