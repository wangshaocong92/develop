export PROJECT_PATH=$(cd $(dirname $0)/..; pwd)
export TRT_COOKBOOK_PATH=$PROJECT_PATH"/doc/trt_samples/cookbook"
export PYTHONPATH=$PROJECT_PATH"/doc/trt_samples/cookbook:$PYTHONPATH"
