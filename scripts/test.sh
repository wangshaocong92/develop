#!/bin/bash
source $(cd $(dirname $0)/../; pwd)/scripts/env.sh


export BUILD_PATH=$PROJECT_PATH/build


LOG_DIR="${PROJECT_PATH}/test_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="ctest_${TIMESTAMP}.log"

# 执行CTest测试
run_ctest() {
     local test_target="${1:-默认测试}"
     file=${LOG_DIR}/"$test_target"_"$LOG_FILE"
     echo "日志文件: $file"
   # 执行CTest并双路输出
    {
        echo "===== 测试环境 $test_target ====="
        echo "时间: $(date)"
        echo "目录: $(pwd)"
        echo "CTest版本: $(ctest --version)"
        echo "==================="

        # 核心测试命令
        ctest --output-on-failure --rerun-failed  --verbose
        exit_code=$?
        echo "测试退出码: $exit_code"
    } | tee "$file"   # 同时输出到终端和文件

    return $exit_code
}


# 创建日志目录
mkdir -p "$LOG_DIR"

if [ "$1" = "all" ]; then
   export TEST_ALL=1
fi

if [ ! -d $BUILD_PATH ]; then
   echo "build path not exist, please build first"
   exit 1
fi

cd $BUILD_PATH

if [[ "$1" = "kernel" || $TEST_ALL -eq 1 ]]; then
    if [ ! -d $BUILD_PATH/kernel ]; then
        echo "kernel test not exist, please build first"
        exit 1
    fi
    cd $BUILD_PATH/kernel
    run_ctest "kernel"
    if [[ $? != 0 ]]; then
        exit 1
    fi
    cd $BUILD_PATH
fi


echo "All tests passed!"