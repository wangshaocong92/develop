#!/bin/bash
source $(cd $(dirname $0)/../; pwd)/scripts/env.sh

export BUILD_PATH=$PROJECT_PATH/build
if [ ! -d $BUILD_PATH ]; then
    mdkir -p $BUILD_PATH
else
    rm -rf $BUILD_PATH/*
fi

cd $BUILD_PATH
conan install $PROJECT_PATH --build=missing
cmake $PROJECT_PATH -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$(pwd) -DCMAKE_VERBOSE_MAKEFILE=ON
make -j$(nproc)