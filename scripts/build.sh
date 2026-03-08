#!/bin/bash -e
source $(cd $(dirname $0)/../; pwd)/scripts/env.sh

export BUILD_PATH=$PROJECT_PATH/build

if [ ! -d $BUILD_PATH ]; then
    mkdir -p $BUILD_PATH
else
    rm -rf $BUILD_PATH/*
fi

export CMAKE_BUILD_PARALLEL_LEVEL=$(awk '/MemTotal/ {mem_gb=$2/1024/1024; j=int(mem_gb/3); if (j<1) j=1; print j}' /proc/meminfo)
echo "建议并行任务数: $CMAKE_BUILD_PARALLEL_LEVEL"

echo $PATH
cd $BUILD_PATH
conan install $PROJECT_PATH -c tools.build:jobs=$CMAKE_BUILD_PARALLEL_LEVEL --build=missing --output-folder=.
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$(pwd) -DCUDAToolkit_ROOT=/usr/local/cuda-12 -DCMAKE_VERBOSE_MAKEFILE=ON  -DCMAKE_INSTALL_PREFIX=$PROJECT_PATH/install $PROJECT_PATH
ninja -j$(nproc)
ninja install

cp $PROJECT_PATH/build/compile_commands.json $PROJECT_PATH/