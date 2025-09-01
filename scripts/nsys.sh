#!/bin/bash
source $(cd $(dirname $0)/../; pwd)/scripts/env.sh
echo $1
nsys profile --trace=cuda,nvtx,osrt --stats=true --cuda-memory-usage=true \
             --cuda-um-cpu-page-faults=true --cuda-event-trace=false \
             -o profile_result $1