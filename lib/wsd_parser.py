'''
Utilities for parsing standardized XML/key format from
Raganato et al's WSD Evaluation Framework.
'''

from bs4 import BeautifulSoup
import codecs

def readLabels(f):
    lbls, unique_sense_IDs = {}, set()
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            (word_ident, *sense_labels) = [s.strip() for s in line.split()]
            lbls[word_ident] = sense_labels
            for lbl in sense_labels:
                unique_sense_IDs.add(lbl)
    return lbls, unique_sense_IDs

def processSentences(xmlf, get_lemmas=False):
    with codecs.open(xmlf, 'r', 'utf-8') as stream:
        corpus_xml = stream.read()
    soup = BeautifulSoup(corpus_xml, 'lxml-xml')

    orig_sentences = soup.find_all('sentence')
    sentences_words = []
    sentences_instances = []

    for sent in orig_sentences:
        sentence_words = []
        indexed_instances = []

        children = list(sent.children)
        for child in children:
            if child != '\n':
                if child.name == 'instance':
                    instance = [
                        child.get('id'),
                        len(sentence_words)
                    ]
                    if get_lemmas:
                        instance.append(child.get('lemma'))
                    indexed_instances.append(instance)
                sentence_words.append(child.text)

        sentences_words.append(sentence_words)
        sentences_instances.append(indexed_instances)

    return (sentences_words, sentences_instances)
