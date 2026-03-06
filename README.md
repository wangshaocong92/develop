# 开发环境
此项目使用docker作为开发环境

## docker 
### 镜像要求
* 显卡支持 cuda12及以上版本
### 镜像的构建
请在文件的同级目录执行```docker build -t ${your-name}:latest .```。此命令会生成一个此项目所需的镜像

### 启动镜像
```docker run -itd --gpus all -v ${work path}:/home/wsc/workspace ${your-name} zsh```