#!/bin/bash
# 脚本：compile_latex_from_exe.sh
# 功能：运行指定的 exe，将其 stdout 作为 LaTeX 源码，用 pdflatex 编译生成 PDF

set -e  # 遇到错误立即退出（可选项，根据需要注释掉）

# 显示使用方法
usage() {
    echo "Usage: $0 <path_to_exe>"
    exit 1
}

# 检查参数个数
if [ $# -ne 1 ]; then
    usage
fi

EXE="$1"

# 检查可执行文件是否存在且可执行
if [ ! -x "$EXE" ]; then
    echo "错误：'$EXE' 不存在或不可执行。"
    exit 1
fi

# 检查 pdflatex 是否可用
if ! command -v pdflatex &> /dev/null; then
    echo "错误：未找到 pdflatex。请安装 TeX Live："
    echo "  sudo apt update && sudo apt install texlive texlive-latex-extra"
    exit 1
fi

# 创建临时目录用于存放所有中间文件，保持工作区整洁
WORK_DIR=$(mktemp -d)
trap "rm -rf \"$WORK_DIR\"" EXIT  # 脚本退出时自动清理临时目录

# 生成 LaTeX 源文件名
TEX_FILE="$WORK_DIR/output.tex"

# 运行 exe，将标准输出写入 .tex 文件
echo "正在运行 '$EXE' 并捕获 LaTeX 源码..."
if ! "$EXE" > "$TEX_FILE"; then
    echo "警告：可执行文件运行失败，但已尝试捕获输出。"
fi

# 检查生成的 .tex 文件是否为空
if [ ! -s "$TEX_FILE" ]; then
    echo "错误：生成的 LaTeX 源码为空。"
    exit 1
fi

# 进入工作目录编译，避免在当前目录产生辅助文件
cd "$WORK_DIR"

# 运行 pdflatex 编译（首次）
echo "正在编译 LaTeX 源码..."
pdflatex -interaction=nonstopmode output.tex > /dev/null 2>&1 || true

# 如果文档包含引用、目录等，可能需要多次编译。简单起见，再编译一次确保引用更新
pdflatex -interaction=nonstopmode output.tex > /dev/null 2>&1 || true

# 检查 PDF 是否生成
if [ -f output.pdf ]; then
    # 将生成的 PDF 复制到当前工作目录（调用脚本时所在的目录）
    cp output.pdf "$OLDPWD/$(basename "$EXE" .exe).pdf"
    echo "成功：PDF 已生成 -> $OLDPWD/$(basename "$EXE" .exe).pdf"
else
    echo "错误：编译失败，请检查日志：$WORK_DIR/output.log"
    # 可选择性保留日志，这里显示日志内容
    cat output.log
    exit 1
fi