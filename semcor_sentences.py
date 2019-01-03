import codecs
import math
import numpy as np
import tensorflow as tf
import configparser
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
import pyemblib
import configlogger
from drgriffis.common import log
from lib import wsd_parser
from lib.elmo_runner import ELMoParams, ELMoRunner, prepVocabulary

def getELMoRepresentations(sentences_words, sentences_instances, semcor_labels,
        unique_sense_IDs, bilm_params):

    sense_embeddings = {}
    for sense_ID in unique_sense_IDs:
        sense_embeddings[sense_ID] = []

    with tf.Session() as sess:
        log.writeln('  (1) Setting up ELMo')
        elmo = ELMoRunner(sess, bilm_params)

        # batch up the data
        sentence_ids = elmo.preprocess(sentences_words)

        batch_size = 25
        num_batches = math.ceil(sentence_ids.shape[0] / batch_size)
        batch_start = 0
        log.writeln('  (2) Extracting sense embeddings from sentences')
        log.track(message='    >> Processed {0}/{1:,} batches'.format('{0:,}',num_batches), writeInterval=5)
        while batch_start < sentence_ids.shape[0]:
            batch_sentence_ids = sentence_ids[batch_start:batch_start + batch_size]
            elmo_sentence_input_ = elmo(batch_sentence_ids)

            for i in range(elmo_sentence_input_.shape[0]):
                sentence_indices = sentences_instances[batch_start+i]
                for (instance_ID, ix) in sentence_indices:
                    senses = semcor_labels[instance_ID]
                    for sense in senses:
                        sense_embeddings[sense].append(
                            elmo_sentence_input_[i][ix]
                        )

            log.tick()
            batch_start += batch_size
    log.flushTracker()

    log.writeln('  (3) Calculating mean per-sense embeddings')
    mean_sense_embeddings = pyemblib.Embeddings()
    for (sense_ID, embedding_list) in sense_embeddings.items():
        if len(embedding_list) > 0:
            mean_sense_embeddings[sense_ID] = np.mean(embedding_list, axis=0)
        else:
            log.writeln('[WARNING] Sense ID "%s" found no embeddings' % sense_ID)
    
    return mean_sense_embeddings

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
            ('Vocab', config['SemCor']['Vocab']),
        ]),
        ('ELMo', [
            ('Weights', config['ELMo']['Weights']),
            ('Options', config['ELMo']['Options']),
        ]),
        ('Output file', config['SemCor']['Embeddings']),
    ])

    t_sub = log.startTimer('Reading SemCor labels from %s...' % config['SemCor']['Labels'])
    semcor_labels, unique_sense_IDs = wsd_parser.readLabels(config['SemCor']['Labels'])
    log.stopTimer(t_sub, message='Read {0:,} labels ({1:,} unique senses) in {2}s.\n'.format(
        len(semcor_labels), len(unique_sense_IDs), '{0:.2f}'
    ))

    t_sub = log.startTimer('Pre-processing SemCor text from %s...' % config['SemCor']['XML'])
    (sentences_words, sentences_instances) = wsd_parser.processSentences(config['SemCor']['XML'])
    log.stopTimer(t_sub, message='Read {0:,} sentences in {1}s.\n'.format(
        len(sentences_words), '{0:.2f}'
    ))

    log.writeln('Pre-processing SemCor vocabulary...')
    max_char_len = prepVocabulary(sentences_words, config['SemCor']['Vocab'])
    log.writeln('Wrote vocabulary to {0}.\nMax character length: {1:,}\n'.format(
        config['SemCor']['Vocab'], max_char_len
    ))

    log.writeln('OVERRIDING max_char_len to 50!\n')
    max_char_len = 50

    bilm_params = ELMoParams()
    bilm_params.options_file = config['ELMo']['Options']
    bilm_params.weights_file = config['ELMo']['Weights']
    bilm_params.vocab_file = config['SemCor']['Vocab']
    bilm_params.max_char_len = max_char_len

    t_sub = log.startTimer('Getting ELMo representations for SemCor senses...')
    sense_embeddings = getELMoRepresentations(
        sentences_words,
        sentences_instances,
        semcor_labels,
        unique_sense_IDs,
        bilm_params
    )
    log.stopTimer(t_sub, message='Calculated embeddings for {0:,} senses in {1}s.\n'.format(
        len(sense_embeddings), '{0:.2f}'
    ))

    t_sub = log.startTimer('Writing sense embeddings to %s...' % config['SemCor']['Embeddings'])
    pyemblib.write(sense_embeddings, config['SemCor']['Embeddings'])
    log.stopTimer(t_sub, message='Completed writing embeddings in {0:.2f}s.\n')
    
    log.stop()
