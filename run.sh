#!/bin/bash
#
# Run full reimplementation of WSD experiments.
#  (1) Create embeddings of senses from SemCor
#  (2) Extract all lemmas from SemCor
#  (3) Generate embedded features for WSD Evaluation Framework samples
#  (4) Run nearest neighbor prediction with backoff to WN First Sense baseline
#
# If any of steps (1)-(3) have previously been completed, they will be skipped.
#
# Requires having run ./setup.sh first.

JAVAC=javac
JAVA=java

if [ ! -e dependencies/pyconfig.sh ]; then
    echo "Please run ./setup.sh before running this script!"
    exit
else
    source dependencies/pyconfig.sh
fi

SemCorEmbeddings=$(${PY} -m lib.configgetter config.ini SemCor Embeddings)
SemCorLemmas=$(${PY} -m lib.configgetter config.ini SemCor Lemmas)
EvalMentions=data/wsd_eval_framework.mentions
EvalMentionMap=data/wsd_eval_framework.mention_map
EvalPredictions=data/wsd_eval_framework.predictions
WNFSBaselinePredictions=$(${PY} -m lib.configgetter config.ini Experiment WNFSBaselinePredictions)
Scorer=data/WSD_Evaluation_Framework/Evaluation_Datasets/Scorer.class

if [ ! -e "${SemCorEmbeddings}" ]; then
    echo "-----------------------------------------"
    echo "Generating SemCor sense embeddings"
    echo "-----------------------------------------"
    ${PY} -m semcor_sentences \
        config.ini \
        -l data/logs/semcor_sentences.log
fi

if [ ! -e "${SemCorLemmas}" ]; then
    echo "-----------------------------------------"
    echo "Extracting SemCor lemmas"
    echo "-----------------------------------------"
    ${PY} -m list_semcor_lemmas \
        config.ini \
        -l data/logs/semcor_lemmas.log
fi

if [ ! -e "${EvalMentions}" ]; then
    echo "-----------------------------------------"
    echo "Generating WSD Eval Framework features"
    echo "-----------------------------------------"
    ${PY} -m extract_mentions \
        --config config.ini \
        --wsd-test-only \
        --wsd-dataset-map-file ${EvalMentionMap} \
        -l data/logs/extract_mentions.log \
        ${EvalMentions}
fi

if [ ! -e "${EvalPredictions}" ]; then
    echo "-----------------------------------------"
    echo "Generating WSD Eval Framework predictions"
    echo "-----------------------------------------"
    ${PY} -m wsd_baseline_experiment \
        ${EvalMentions} \
        --mention-map ${EvalMentionMap} \
        --wordnet-baseline-input-predictions ${WNFSBaselinePredictions} \
        --training-lemmas ${SemCorLemmas} \
        --semcor-embeddings ${SemCorEmbeddings} \
        --elmo-baseline-eval-predictions ${EvalPredictions} \
        -l data/logs/wsd_baseline_experiment.log
fi

if [ ! -e "${Scorer}" ]; then
    echo "-----------------------------------------"
    echo "Compiling WSD Eval Framework scorer"
    echo "-----------------------------------------"
    curdir=$(pwd)
    cd data/WSD_Evaluation_Framework/Evaluation_Datasets
    ${JAVAC} Scorer.java
    cd ${curdir}
fi

echo "-----------------------------------------"
echo "Running WSD Eval Framework scorer"
echo "-----------------------------------------"
curdir=$(pwd)
cd data/WSD_Evaluation_Framework/Evaluation_Datasets
java Scorer ALL/ALL.gold.key.txt ${curdir}/${EvalPredictions}
