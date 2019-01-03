'''
'''

import os
import codecs
import numpy as np
import time
from nltk.corpus import wordnet as wn
from datetime import datetime
from drgriffis.common import log
import configlogger
import pyemblib
from lib import mention_file
from lib import dataset_map_utils

def predictionID(ds, instance_ID):
    return ('%s.%s' % (
        dataset_map_utils.DatasetOutputMap[ds],
        instance_ID
    ))

def readTrainingLemmas(f):
    lemmas = set()
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            lemmas.add(line.strip())
    return lemmas

def writeWSDFrameworkPredictions(predictions, mention_map, f):
    with codecs.open(f, 'w', 'utf-8') as stream:
        for (mention_ID, prediction) in predictions:
            (ds, instance_ID) = mention_map[mention_ID][:2]
            stream.write('%s.%s %s\n' % (
                dataset_map_utils.DatasetOutputMap[ds],
                instance_ID,
                prediction
            ))

def loadWSDFrameworkPredictions(f):
    predictions = {}
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            (_id, prediction) = [s.strip() for s in line.split()]
            predictions[_id] = prediction
    return predictions

def getNearestNeighborKey(context_repr, normed_embeddings):
    best_key, min_dist = None, 1e8
    repr_norm = np.linalg.norm(context_repr)
    for (k, normed_v) in normed_embeddings.items():
        # cosine distance
        cos_sim = (
            np.dot(context_repr, normed_v)
            /
            repr_norm
        )
        cos_dist = 1 - cos_sim
        if cos_dist < min_dist:
            best_key = k
            min_dist = cos_dist
    return best_key

def getNearestNeighborKey2(context_repr, normed_embedding_arr, ordered_vocab):
    ## cosine similarity
    #cos_sim_array = np.matmul(
    #    context_repr,
    #    normed_embedding_arr
    #)
    #best_ix = np.argmax(cos_sim_array)
    # Euclidean distance
    sq_euc_dist_array = np.sum(
        (np.transpose(normed_embedding_arr) - context_repr) ** 2,
        axis=1
    )
    best_ix = np.argmin(sq_euc_dist_array)
    best_key = ordered_vocab[best_ix]
    return best_key

def ELMoBaseline(mentions, mention_map, backoff_preds, training_lemmas,
        semcor_embeddings, output_predsf):
    log.writeln('Running ELMo baseline\n')

    # pre-norm the semcor embeddings
    log.writeln('Norming SemCor embeddings...')
    normed_semcor_embeddings = pyemblib.Embeddings()
    for (k,v) in semcor_embeddings.items():
        normed_semcor_embeddings[k] = (
            v / np.linalg.norm(v)
        )
    #semcor_embeddings = normed_semcor_embeddings
    ordered_vocab, semcor_embeddings = normed_semcor_embeddings.toarray()
    semcor_embeddings = np.transpose(semcor_embeddings)
    log.writeln('Done.\n')

    predictions, correct = [], 0
    num_elmo, num_backoff = 0, 0

    log.track(message='  >> Processed {0:,}/%s samples ({1:,} ELMo, {2:,} backoff)' % ('{0:,}'.format(len(mentions))), writeInterval=5)
    for m in mentions:
        (ds, instance_ID, lemma) = mention_map[m.ID]
        if lemma in training_lemmas:
            #prediction = getNearestNeighborKey(m.context_repr, semcor_embeddings)
            prediction = getNearestNeighborKey2(m.context_repr, semcor_embeddings, ordered_vocab)
            num_elmo += 1
        else:
            prediction = backoff_preds[predictionID(ds, instance_ID)]
            num_backoff += 1
        predictions.append( (m.ID, prediction) )
        if prediction == m.CUI:
            correct += 1
        log.tick(num_elmo, num_backoff)
    log.flushTracker(num_elmo, num_backoff)

    writeWSDFrameworkPredictions(predictions, mention_map, output_predsf)
    log.writeln('\n-- ELMo baseline --')
    log.writeln('Accuracy: {0:.4f} ({1:,}/{2:,})\n'.format(
        float(correct)/len(predictions),
        correct,
        len(predictions)
    ))
    log.writeln('# ELMo: {0:,}\n# backoff: {1:,}\n'.format(
        num_elmo,
        num_backoff
    ))

def wordnetFirstSenseBaseline(mentions, mention_map, predsf):
    predictions, correct = [], 0
    for m in mentions:
        # m.candidates is the (ranked) list returned by WordNet
        # Look, I have no idea how Raganato et al. got their list
        # out. But this can be scrapped, because it VASTLY
        # underperforms their FirstSense baseline.
        (_, _, lemma) = mention_map[m.ID]
        synsets = wn.synsets(lemma)
        if lemma == 'peculiar':
            print(synsets)
        found_it = False
        for j in range(len(synsets)):
            this_lemma = synsets[j].lemmas()[0].name()
            if lemma == 'peculiar':
                print(j, this_lemma)
            if this_lemma == lemma:
                found_it = True
                break
        if not found_it:
            j = 0
        guess = synsets[j].lemmas()[0].key()
        if lemma == 'peculiar':
            print(j, guess)
        #guess = wn.synsets(lemma)[0].lemmas()[0].key()
        #guess = m.candidates[0]
        predictions.append( (m.ID, guess) )
        if guess == m.CUI:
            correct += 1

    writeWSDFrameworkPredictions(predictions, mention_map, predsf)
    log.writeln('-- WordNet first sense baseline --')
    log.writeln('Accuracy: {0:.4f} ({1:,}/{2:,})\n'.format(
        float(correct)/len(predictions),
        correct,
        len(predictions)
    ))


if __name__=='__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog MENTIONS [options]',
                description='Runs the LogLinearLinker model using the embeddings in ENTITY_FILE and CTX_FILE'
                            ' on the mentions in MENTIONS.')
        parser.add_option('--mention-map', dest='mention_mapf',
                help='(REQUIRED) file mapping mention IDs to dataset info')
        parser.add_option('-l', '--logfile', dest='logfile',
                help=str.format('name of file to write log contents to (empty for stdout)'),
                default=None)

        group = optparse.OptionGroup(parser, 'ELMo baseline options')
        parser.add_option('--elmo-baseline-eval-predictions', dest='elmo_baseline_eval_predictions',
                help='file to write WSD Evaluation Framework formatted predictions from'
                     ' ELMo baseline with precalculated WordNet first sense backoff')
        group.add_option('--semcor-embeddings', dest='semcor_embf',
                help='entity embedding file (REQUIRED with --elmo-baseline-eval-predictions)')
        group.add_option('--wordnet-baseline-input-predictions', dest='wordnet_baseline_input_predictions',
                help='file with pre-calculated WordNet first sense baseline predictions'
                     ' (REQUIRED with --elmo-baseline-eval-predictions)')
        group.add_option('--training-lemmas', dest='training_lemmasf',
                help='file listing lemmas from SemCor training set'
                     ' (REQUIRED with --elmo-baseline-eval-predictions)')

        group = optparse.OptionGroup(parser, 'WordNet bsaeline options')
        group.add_option('--wordnet-baseline-eval-predictions', dest='wordnet_baseline_eval_predictions',
                help='file to write WSD Evaluation Framework formatted predictions from'
                     ' WordNet first sense baseline')

        (options, args) = parser.parse_args()

        if len(args) != 1 or not options.mention_mapf or (options.elmo_baseline_eval_predictions and (not options.semcor_embf or not options.wordnet_baseline_input_predictions or not options.training_lemmasf)):
            parser.print_help()
            exit()
        (mentionf,) = args
        return mentionf, options

    ## Getting configuration settings
    mentionf, options = _cli()
    log.start(logfile=options.logfile, stdout_also=True)
    configlogger.writeConfig(log, [
        ('Mention file', mentionf),
        ('Mention map file', options.mention_mapf),
        ('WordNet first sense baseline settings', [
            ('Output predictions file', options.wordnet_baseline_eval_predictions),
        ]),
        ('ELMo baseline settings', [
            ('Output predictions file', options.elmo_baseline_eval_predictions),
            ('SemCor embeddings', options.semcor_embf),
            ('Training lemmas file', options.training_lemmasf),
            ('Pre-calculated WN first sense backoff predictions', options.wordnet_baseline_input_predictions),
        ]),
    ], title="ELMo WSD baselines replication")

    t_sub = log.startTimer('Reading mentions from %s...' % mentionf, newline=False)
    mentions = mention_file.read(mentionf)
    log.stopTimer(t_sub, message='Read %d mentions ({0:.2f}s)' % len(mentions))

    log.writeln('Reading mention dataset data from %s...' % options.mention_mapf)
    mention_map = dataset_map_utils.readDatasetMap(options.mention_mapf, get_IDs=True, get_lemmas=True)
    log.writeln('Mapped dataset info for {0:,} mentions.\n'.format(len(mention_map)))

    if options.wordnet_baseline_eval_predictions:
        wordnetFirstSenseBaseline(mentions, mention_map, options.wordnet_baseline_eval_predictions)
    if options.elmo_baseline_eval_predictions:
        log.writeln('Reading set of training lemmas from %s...' % options.training_lemmasf)
        training_lemmas = readTrainingLemmas(options.training_lemmasf)
        log.writeln('Read {0:,} lemmas.\n'.format(len(training_lemmas)))

        log.writeln('Reading SemCor sense embeddings from %s...' % options.semcor_embf)
        semcor_embeddings = pyemblib.read(options.semcor_embf)
        log.writeln('Read embeddings for {0:,} senses.\n'.format(len(semcor_embeddings)))

        log.writeln('Reading backoff predictions from %s...' % options.wordnet_baseline_input_predictions)
        wn_first_sense_preds = loadWSDFrameworkPredictions(options.wordnet_baseline_input_predictions)
        log.writeln('Read predictions for {0:,} samples.\n'.format(len(wn_first_sense_preds)))

        ELMoBaseline(mentions, mention_map, wn_first_sense_preds, training_lemmas,
            semcor_embeddings, options.elmo_baseline_eval_predictions)

    log.stop()
