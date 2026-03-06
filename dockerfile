# 使用 Ubuntu 24.04 作为基础镜像
FROM ubuntu:24.04

# 避免在安装过程中出现交互式提示（如时区选择）
ENV DEBIAN_FRONTEND=noninteractive

# 更新软件包列表并安装基础工具和依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    # 基础工具
    curl \
    wget \
    vim \
    git \
    # 编译工具（如果需要）
    build-essential \
    cmake \
    # 网络工具
    net-tools \
    iputils-ping \
    # 其他常用
    ca-certificates \
    # NVIDIA Container Toolkit 安装所需的依赖
    gnupg \
    lsb-release \
    # ========== Zsh 相关 ==========
    zsh \
    # Oh My Zsh 需要 curl 或 wget（已包含），可选安装其他插件依赖
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置 NVIDIA 相关的环境变量（推荐但非必需）
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# 安装 NVIDIA Container Toolkit
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    && apt-get update \
    && apt-get install -y nvidia-container-toolkit \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 注意：Docker 守护进程的配置需要在宿主机上完成，不能在容器内进行。
# 需要在宿主机执行以下命令（已包含在注释中供参考）：
# sudo nvidia-ctk runtime configure --runtime=docker
# sudo systemctl restart docker
RUN apt-get update && apt-get install -y cuda-toolkit-12-8
# 恢复交互式前端（可选）
ENV DEBIAN_FRONTEND=dialog

# 创建一个非 root 用户（例如 appuser）并赋予 sudo 权限
ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
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

# 设置工作目录
WORKDIR /home/$USERNAME/workspace

# 默认命令（启动 Zsh）
CMD ["/usr/bin/zsh"]