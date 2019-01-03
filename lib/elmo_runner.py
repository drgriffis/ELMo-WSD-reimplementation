'''
Wrapper for running pretrained ELMo on preprocessed sentences.
'''

import codecs
import numpy as np
import tensorflow as tf
from bilm import Batcher, BidirectionalLanguageModel, weight_layers

class ELMoParams:
    
    def __init__(self,
            options_file=None,
            weights_file=None,
            vocab_file=None,
            max_char_len=-1
        ):
        self.options_file = options_file
        self.weights_file = weights_file
        self.vocab_file = vocab_file
        self.max_char_len = max_char_len


class ELMoRunner:
    
    def __init__(self, session, bilm_params):
        self.params = bilm_params

        # Create a Batcher to map text to character ids.
        self.batcher = Batcher(self.params.vocab_file, self.params.max_char_len)

        # Input placeholders to the biLM.
        self.sentence_character_ids = tf.placeholder('int32', shape=(None, None, self.params.max_char_len))

        # Build the biLM graph.
        bilm = BidirectionalLanguageModel(
            self.params.options_file,
            self.params.weights_file,
        )

        # Get ops to compute the LM embeddings.
        sentence_embeddings_op = bilm(self.sentence_character_ids)

        self.elmo_sentence_input = weight_layers(
            'input',
            sentence_embeddings_op,
            l2_coef=0.0,
            use_top_only=True
        )

        self.sess = session
        self.sess.run(tf.global_variables_initializer())

    def preprocess(self, sentences_words):
        return self.batcher.batch_sentences(sentences_words)

    def __call__(self, batch_sentence_ids):
        (elmo_sentence_input_,) = self.sess.run(
            [self.elmo_sentence_input['weighted_op']],
            feed_dict={self.sentence_character_ids: batch_sentence_ids}
        )
        return elmo_sentence_input_

def prepVocabulary(sentences_words, vocabf):
    all_vocab = set(['<S>', '</S>'])
    max_len = 0
    for sentence in sentences_words:
        for word in sentence:
            all_vocab.add(word)
            if len(word) > max_len:
                max_len = len(word)

    with codecs.open(vocabf, 'w', 'utf-8') as stream:
        for word in all_vocab:
            stream.write('%s\n' % word)

    return max_len
