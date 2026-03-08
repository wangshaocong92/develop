export PROJECT_PATH=$(cd $(dirname $0)/..; pwd)
export CUDA_HOME=/usr/local/cuda-12 
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH