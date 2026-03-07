# 开发环境
此项目使用docker作为开发环境

## docker 
### 镜像要求
* 显卡支持 cuda12及以上版本

### 镜像的构建

请在文件的同级目录执行```docker build -t wsc_ubuntu24.04  .  --build-arg USERNAME=$(whoami) --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g)```。此命令会生成一个此项目所需的镜像

### 启动镜像
```docker run -itd --gpus all --user $(whoami) -v ${work path}:/home/$(whoami)/workspace -v ${user conan pkg path}:/home/$(whoami)/.conan2/p ${your-name} zsh```
eg.
```docker run -itd --gpus all --user wsc -v /home/wsc-machine/code/develop:/home/wsc/workspace -v /home/wsc-machine/.conan2/p:/home/wsc/.conan2/p wsc-ubuntu24.04  zsh ```