'''
Data access wrappers for WSD datasets in Raganato et al
WSD Evaluation Framework project
'''

from lib import wsd_parser

class WSDDataset:
    def __init__(self, config, name):
        self.config = config
        self.name = name

        (
            self.labels,
            self.unique_sense_IDs
        ) = wsd_parser.readLabels(self.config[name]['Labels'])

        (
            self.sentences_words,
            self.sentences_instances
        ) = wsd_parser.processSentences(self.config[name]['XML'], get_lemmas=True)

class SemCor(WSDDataset):
    def __init__(self, config):
        super().__init__(config, 'SemCor')

class SemEval2007(WSDDataset):
    def __init__(self, config):
        super().__init__(config, 'SemEval2007')

class SemEval2013(WSDDataset):
    def __init__(self, config):
        super().__init__(config, 'SemEval2013')

class SemEval2015(WSDDataset):
    def __init__(self, config):
        super().__init__(config, 'SemEval2015')

class SensEval2(WSDDataset):
    def __init__(self, config):
        super().__init__(config, 'SensEval2')

class SensEval3(WSDDataset):
    def __init__(self, config):
        super().__init__(config, 'SensEval3')

class EvalAll(WSDDataset):
    def __init__(self, config):
        super().__init__(config, 'EvalAll')

def allAsList(config, test_only=False):
    if test_only:
        lst = []
    else:
        lst = [SemCor(config)]
    lst.extend([
        SemEval2007(config),
        SemEval2013(config),
        SemEval2015(config),
        SensEval2(config),
        SensEval3(config)
    ])
    return lst
