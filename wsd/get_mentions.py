import math
import optparse
from lib import mention_file
from nltk.corpus import wordnet as wn
from drgriffis.common import preprocessing, log
import tensorflow as tf
from lib.elmo_runner import ELMoParams, ELMoRunner, prepVocabulary

def getEmbeddedSingleMention(lemma, context_repr, label, ID):
    # grab the candidates
    candidates = [
        synset.lemmas()[0].key()
            for synset in wn.synsets(lemma)
    ]

    return mention_file.EmbeddedMention(
        label,
        None,
        context_repr,
        candidates,
        ID=ID
    )

def getAllMentions(datasets, log=log, mention_map_file=None):
    ds_map = {}

    # pre-generate the vocabulary of all datasets
    all_sentences = []
    for ds in datasets:
        all_sentences.extend(ds.sentences_words)
    prepVocabulary(all_sentences, datasets[0].config['Experiment']['TotalVocab'])

    params = ELMoParams(
        options_file=datasets[0].config['ELMo']['Options'],
        weights_file=datasets[0].config['ELMo']['Weights'],
        vocab_file=datasets[0].config['Experiment']['TotalVocab'],
        max_char_len=int(datasets[0].config['ELMo']['MaxCharLen']),
    )
    elmo_batch_size = int(datasets[0].config['ELMo']['BatchSize'])

    sess = tf.Session()
    elmo = ELMoRunner(sess, params)

    samples = []
    for ds in datasets:
        log.writeln('\nProcessing dataset %s...' % ds.name)
        _getELMoMentions(
            ds.sentences_words,
            ds.sentences_instances,
            ds.labels,
            ds.name,
            samples,
            ds_map,
            elmo,
            batch_size=elmo_batch_size
        )

    if mention_map_file:
        with open(mention_map_file, 'w') as stream:
            for (mention_ID, (ds_name, instance_ID, lemma)) in ds_map.items():
                stream.write('%d\t%s\t%s\t%s\n' % (mention_ID, ds_name, instance_ID, lemma))
                
    return samples

def _getELMoMentions(sentences_words, sentences_instances, labels, ds_name,
        samples, ds_map, elmo, batch_size=25):
    sentence_ids = elmo.preprocess(sentences_words)

    batch_start = 0
    num_batches = math.ceil(sentence_ids.shape[0] / batch_size)

    log.track(message='    >> Processed {0}/{1:,} batches'.format('{0:,}',num_batches), writeInterval=5)
    while batch_start < sentence_ids.shape[0]:
        batch_sentence_ids = sentence_ids[batch_start:batch_start + batch_size]
        elmo_sentence_input_ = elmo(batch_sentence_ids)

        for i in range(elmo_sentence_input_.shape[0]):
            sentence_indices = sentences_instances[batch_start+i]
            for (instance_ID, ix, lemma) in sentence_indices:
                # instance may have multiple correct senses;
                # since this is just for training and we need one hot label,
                # just take the first one
                label = labels[instance_ID][0]
                context_repr = elmo_sentence_input_[i][ix]
                samples.append(getEmbeddedSingleMention(
                    #sentences_words[batch_start+i][ix],
                    lemma,
                    context_repr,
                    label,
                    ID=len(samples)
                ))
                ds_map[samples[-1].ID] = (ds_name, instance_ID, lemma)
        log.tick()
        batch_start += batch_size
    log.flushTracker()
