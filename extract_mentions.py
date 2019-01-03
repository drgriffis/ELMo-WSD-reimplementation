'''
'''

import numpy as np
import configparser
import configlogger
from lib import mention_file
import wsd
from drgriffis.common import log, util

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog [options] OUTFILE',
                description='Generates common-format file giving mentions (with contextual information) for full dataset')
        parser.add_option('--config', dest='dataset_configf',
                default='config.ini',
                help='path to dataset configuration file (default: %default)')
        parser.add_option("--wsd-dataset-map-file", dest="wsd_mention_map_file",
                help='file to save map from mention ID to WSD dataset (default: unused)')
        parser.add_option('--embed-with-elmo', dest='wsd_embed_with_elmo',
                action='store_true', default=False,
                help='use ELMo to embed the ambiguous words using the full context sentence')
        parser.add_option('--wsd-test-only', dest='wsd_test_only',
                action='store_true', default=False,
                help='only get mentions for test sets (no SemCor)')
        parser.add_option('-l', '--logfile', dest='logfile',
                help='name of file to write log contents to (empty for stdout)',
                default=None)
        (options, args) = parser.parse_args()
        if len(args) != 1:
            parser.print_help()
            exit()
        (outf,) = args
        return outf, options

    outf, options = _cli()
    log.start(logfile=options.logfile)
    configlogger.writeConfig(log, [
        ('Dataset configuration file', options.dataset_configf),
        ('Mention ID->dataset map file', options.wsd_mention_map_file),
        ('Mentions for test data only', options.wsd_test_only),
    ], title='Mention extraction for entity linking')

    config = configparser.ConfigParser()
    config.read(options.dataset_configf)

    t_sub = log.startTimer('Generating WSD Evaluation Framework features.')
    datasets = wsd.allAsList(config, test_only=options.wsd_test_only)
    mentions = wsd.getAllMentions(datasets,log=log,
        mention_map_file=options.wsd_mention_map_file)
    log.stopTimer(t_sub, 'Extracted %d samples.' % len(mentions))

    t_sub = log.startTimer('Writing samples to %s...' % outf, newline=False)
    mention_file.write(mentions, outf)
    log.stopTimer(t_sub, message='Done ({0:.2f}s).')

    log.stop()
