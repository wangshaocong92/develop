# 使用官方 CUDA 运行时镜像作为基础
FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04

# 避免安装时出现交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 设置 NVIDIA 相关环境变量（便于容器运行时识别 GPU）
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 更新软件包列表并安装基础工具和 Zsh
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    vim \
    git \
    build-essential \
    cmake \
    net-tools \
    iputils-ping \
    ca-certificates \
    gnupg \
    lsb-release \
    zsh \
    sudo \
    clangd \
    clang \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 恢复交互式前端（可选）
ENV DEBIAN_FRONTEND=dialog

# 创建普通用户 wsc（不指定 UID/GID，避免与系统冲突）
ARG USERNAME=wsc

RUN  useradd -m -s /usr/bin/zsh $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    # ========== 设置 Zsh 为默认 Shell ==========
    && chsh -s /usr/bin/zsh $USERNAME \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ========== 可选：安装 Oh My Zsh ==========
# 这步以非 root 用户身份执行，安装到用户家目录
USER $USERNAME
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    # 可选：自定义主题或插件，例如：
    # sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="agnoster"/g' ~/.zshrc
    # git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions
    # sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions)/g' ~/.zshrc
    && echo "source ~/.zshrc" >> ~/.bashrc  # 为兼容性，进入 bash 时也加载 zsh 配置（可选）

# 设置用户家目录环境变量
ENV HOME=/home/$USERNAME

# 安装 Miniforge3 到用户家目录
ENV MINIFORGE_VERSION=24.11.3-0
ENV MINIFORGE_URL=https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh
ENV CONDA_DIR=${HOME}/miniforge3

# 从网络上获取
# RUN wget --progress=bar:force ${MINIFORGE_URL} -O ${HOME}/miniforge.sh \
#     && bash ${HOME}/miniforge.sh -b -p ${CONDA_DIR} \
#     && rm ${HOME}/miniforge.sh \
#     && ${CONDA_DIR}/bin/conda clean --all -y
# 使用本地文件（如果已经下载好了 Miniforge3 安装脚本）
COPY Miniforge3-24.11.3-0-Linux-x86_64.sh ${HOME}/miniforge.sh
RUN  bash ${HOME}/miniforge.sh -b -p ${CONDA_DIR} \
    && rm ${HOME}/miniforge.sh \
    && ${CONDA_DIR}/bin/conda clean --all -y
# 将 conda 加入 PATH
ENV PATH=${CONDA_DIR}/bin:${PATH}

# 安装 conan（使用 pip）
RUN pip install --no-cache-dir conan

# 生成默认 conan profile
RUN conan profile detect


# 可选：创建并激活一个默认的 conda 环境（例如从 environment.yml）
# COPY environment.yml ${HOME}/environment.yml
# RUN conda env create -f ${HOME}/environment.yml
# ENV PATH=${CONDA_DIR}/envs/myenv/bin:$PATH

# 设置工作目录
WORKDIR ${HOME}/workspace

# 默认启动 Zsh
CMD ["/usr/bin/zsh"]