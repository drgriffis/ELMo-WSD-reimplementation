#!/bin/bash
#
# Initial setup for running reimplementation scripts
#
# Downloads dependencies and data as needed.

if [ ! -d dependencies ]; then
    mkdir dependencies
fi

if [ ! -d data ]; then
    mkdir data
fi
if [ ! -d data/logs ]; then
    mkdir data/logs
fi


######################################################
### Dependencies/Python configuration ################

PYCONFIG=dependencies/pyconfig.sh

if [ -e ${PYCONFIG} ]; then
    echo "Python configuration file ${PYCONFIG} exists, skipping"
    source ${PYCONFIG}
else
    # start Python configuration file
    echo "#!/bin/bash" > ${PYCONFIG}

    # Configure Python installation to use
    echo "Python environment to execute with (should include tensorflow)"
    read -p "Path [python3]: " PY
    if [ -z "${PY}" ]; then
        PY=python3
    fi
    echo "export PY=${PY}" >> ${PYCONFIG}

    echo "Checking for dependencies..."

    # check for pyemblib
    ${PY} -c "import pyemblib" 2>>/dev/null
    if [[ $? = 1 ]]; then
        echo
        echo "Cloning pyemblib..."
        cd dependencies
        git clone https://github.com/drgriffis/pyemblib.git
        cd ../
        echo "export PYTHONPATH=\${PYTHONPATH}:$(pwd)/dependencies/pyemblib" >> ${PYCONFIG}
    fi

    # check for configlogger
    ${PY} -c "import configlogger" 2>>/dev/null
    if [[ $? = 1 ]]; then
        echo
        echo "Cloning configlogger..."
        cd dependencies
        git clone https://github.com/drgriffis/configlogger.git
        cd ../
        echo "export PYTHONPATH=\${PYTHONPATH}:$(pwd)/dependencies/configlogger" >> ${PYCONFIG}
    fi

    # check for drgriffis.common
    ${PY} -c "import drgriffis.common" 2>>/dev/null
    if [[ $? = 1 ]]; then
        echo
        echo "Cloning miscutils (drgriffis.common)..."
        cd dependencies
        git clone https://github.com/drgriffis/miscutils.git
        cd ../
        echo "export PYTHONPATH=\${PYTHONPATH}:$(pwd)/dependencies/miscutils/py" >> ${PYCONFIG}
    fi

    # check for bilm
    ${PY} -c "import bilm" 2>>/dev/null
    if [[ $? = 1 ]]; then
        echo
        echo "Cloning bilm-tf..."
        cd dependencies
        git clone https://github.com/allenai/bilm-tf.git
        cd ../
        echo "export PYTHONPATH=\${PYTHONPATH}:$(pwd)/dependencies/bilm-tf" >> ${PYCONFIG}
    fi

    echo
    echo "Python configuration complete."
    echo "Configuration written to ${PYCONFIG}"
fi

echo
echo "Checking for required data..."

######################################################
### Data downloads ###################################

ELMoWeights=$(${PY} -m lib.configgetter config.ini ELMo Weights)
ELMoOptions=$(${PY} -m lib.configgetter config.ini ELMo Options)

if [ ! -e ${ELMoWeights} ]; then
    echo Downloading ELMo weights file...
    cd data
    wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
    echo
    cd ../
else
    echo ELMo weights file found!
fi

if [ ! -e ${ELMoOptions} ]; then
    echo Downloading ELMo options file...
    cd data
    wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json
    echo
    cd ../
else
    echo ELMo options file found!
fi

if [ ! -d data/WSD_Evaluation_Framework ]; then
    echo Downloading WSD Evaluation framework...
    cd data
    wget http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip
    unzip WSD_Evaluation_Framework.zip
    echo
else
    echo WSD Evaluation Framework found!
fi

echo
echo "Setup complete!"
echo "To run WSD experiment, call ./run.sh"
