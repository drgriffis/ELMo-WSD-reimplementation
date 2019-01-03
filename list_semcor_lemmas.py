import codecs
import configparser
import configlogger
from drgriffis.common import log
from lib import wsd_parser

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog CONFIG')
        parser.add_option('-l', '--logfile', dest='logfile',
                help='name of file to write log contents to (empty for stdout)',
                default=None)
        (options, args) = parser.parse_args()
        if len(args) != 1:
            parser.print_help()
            exit()
        return args, options
    (configf,), options = _cli()
    config = configparser.ConfigParser()
    config.read(configf)

    log.start(logfile=options.logfile)
    configlogger.writeConfig(log, [
        ('SemCor', [
            ('XML', config['SemCor']['XML']),
            ('Labels', config['SemCor']['Labels']),
        ]),
        ('Output file', config['SemCor']['Lemmas']),
    ])

    t_sub = log.startTimer('Pre-processing SemCor text from %s...' % config['SemCor']['XML'])
    (sentences_words, sentences_instances) = wsd_parser.processSentences(
        config['SemCor']['XML'], get_lemmas=True)
    log.stopTimer(t_sub, message='Read {0:,} sentences in {1}s.\n'.format(
        len(sentences_words), '{0:.2f}'
    ))

    log.writeln('Collecting set of SemCor lemmas...')
    lemmas = set()
    for sentence_instances in sentences_instances:
        for (instance_ID, ix, lemma) in sentence_instances:
            lemmas.add(lemma)
    log.writeln('Found {0:,} distinct lemmas.\n'.format(len(lemmas)))

    log.writeln('Writing list of lemmas to %s...' % config['SemCor']['Lemmas'])
    with codecs.open(config['SemCor']['Lemmas'], 'w', 'utf-8') as stream:
        for lemma in lemmas:
            stream.write('%s\n' % lemma)
    log.writeln('Done.\n')
    
    log.stop()
