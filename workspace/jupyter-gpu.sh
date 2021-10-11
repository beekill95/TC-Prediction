#!/bin/bash

# Source our .bashrc, should be done automatically, but somehow don't
source $HOME/.bashrc

# Activate conda environment with tensorflow-gpu
conda activate tc_prediction

# Then, forward port back to h2
port=8888
ssh -N -f -R $port:localhost:$port h2

# Finally, start jupyter notebook at the specified port.
jupyter notebook --no-browser --ip 0.0.0.0 --port $port
