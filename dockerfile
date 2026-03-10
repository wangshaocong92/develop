# 使用官方 CUDA 运行时镜像作为基础
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

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
    ninja-build \
    openssh-server \
    tmux \
    texlive-full \
    pkg-config \
    clang-format \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 恢复交互式前端（可选）
ENV DEBIAN_FRONTEND=dialog

# 创建普通用户 wsc（不指定 UID/GID，避免与系统冲突）
ARG USERNAME=wsc
ARG USER_UID=1000
ARG USER_GID=1000


ARG GIT_USER_NAME
ARG GIT_USER_EMAIL


RUN set -ex; \
    # 处理组：确保存在 GID=$USER_GID 且组名为 $USERNAME
    if getent group $USER_GID >/dev/null; then \
        current_group=$(getent group $USER_GID | cut -d: -f1); \
        if [ "$current_group" != "$USERNAME" ]; then \
            groupmod -n $USERNAME $current_group; \
        fi; \
    else \
        groupadd --gid $USER_GID $USERNAME; \
    fi; \
    # 处理用户：确保存在 UID=$USER_UID 且用户名为 $USERNAME
    if id -u $USER_UID >/dev/null 2>&1; then \
        current_user=$(getent passwd $USER_UID | cut -d: -f1); \
        if [ "$current_user" != "$USERNAME" ]; then \
            # 修改用户名，并移动家目录到 /home/$USERNAME
            usermod -l $USERNAME -d /home/$USERNAME -m $current_user; \
        fi; \
    else \
        useradd --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi; \
    # 安装 sudo（如果基础镜像中没有）
    apt-get update && apt-get install -y sudo; \
    # 赋予用户免密 sudo 权限
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME; \
    chmod 0440 /etc/sudoers.d/$USERNAME; \
    # 设置默认 shell 为 zsh（需确保 zsh 已安装，你之前的步骤已安装）
    chsh -s /usr/bin/zsh $USERNAME; \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN  apt-get update \
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
RUN sh -c "$(curl -fsSL https://gh-proxy.org/https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    # 可选：自定义主题或插件，例如：
    # sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="agnoster"/g' ~/.zshrc
    # git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions
    # sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions)/g' ~/.zshrc
    && echo "source ~/.zshrc" >> ~/.bashrc  # 为兼容性，进入 bash 时也加载 zsh 配置（可选）

# 设置用户家目录环境变量
ENV HOME=/home/$USERNAME

# 安装 Miniforge3 到用户家目录
ENV MINIFORGE_VERSION=24.11.3-0
ENV MINIFORGE_URL=https://gh-proxy.org/https://github.com/conda-forge/miniforge/releases/download/24.11.3-0/Miniforge3-24.11.3-0-Linux-x86_64.sh
ENV CONDA_DIR=${HOME}/miniforge3


# 使用本地文件（如果已经下载好了 Miniforge3 安装脚本）
# COPY Miniforge3-24.11.3-0-Linux-x86_64.sh ${HOME}/miniforge.sh
# 从网络上获取
RUN wget --progress=bar:force ${MINIFORGE_URL} -O ${HOME}/miniforge.sh
RUN bash ${HOME}/miniforge.sh -b -p ${CONDA_DIR} \
    && rm ${HOME}/miniforge.sh \
    && ${CONDA_DIR}/bin/conda clean --all -y
    
# 将 conda 加入 PATH
ENV PATH=${CONDA_DIR}/bin:${PATH}

# 安装 conan（使用 pip）
RUN pip install --no-cache-dir conan


# 生成默认 conan profile
RUN conan profile detect

RUN git config --global user.name $GIT_USER_NAME
RUN git config --global user.email $GIT_USER_EMAIL


# 可选：创建并激活一个默认的 conda 环境（例如从 environment.yml）
# COPY environment.yml ${HOME}/environment.yml
# RUN conda env create -f ${HOME}/environment.yml
# ENV PATH=${CONDA_DIR}/envs/myenv/bin:$PATH

# 设置工作目录
WORKDIR ${HOME}/workspace

# 默认启动 Zsh
CMD ["/usr/bin/zsh"]