#!/bin/bash

MC2=$HOME/miniconda2/bin

$MC2/conda create -n pan-lightcurve python=2.7
source $MC2/activate pan-lightcurve
conda install pyqt
pip install -Ur requirements.txt