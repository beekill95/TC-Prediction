#!/bin/bash -i

# Activate conda environment with tensorflow-gpu
conda activate tc_updated_2
# conda activate iub
module load deeplearning

# Then, forward port back to h2
server=${1-h2}
port=${2-8888}
echo "Logging into $server to forward port $port:"
ssh -N -f -R $port:localhost:$port $server

# Finally, start jupyter notebook at the specified port.
echo "Starting jupyter notebook at port $port"
python --version
jupyter-lab --no-browser --ip 0.0.0.0 --port $port --notebook-dir=.
