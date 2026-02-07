# Dockerfile for train_gimbal_curriculum.py
# Provides a complete environment for running the gimbal training script
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Build arguments for user configuration
ARG UID=1006
ARG GID=1006
ARG USERNAME=agent
ARG PASSWORD=rl

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MKL_SERVICE_FORCE_INTEL=1 \
    MUJOCO_GL=egl \
    DISPLAY=:1 \
    XAUTHORITY=/tmp/vnc-share/.Xauthority

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    nano \
    sudo \
    # OpenGL/rendering dependencies for PyBullet
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev \
    # X11 and GUI libraries (for PyBullet visualization if needed)
    xvfb \
    x11-utils \
    x11-apps \
    xauth \
    # Video encoding/ffmpeg for imageio
    ffmpeg \
    # Additional utilities
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    # Fonts for PIL ImageDraw
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set python3.10 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Create user with specified UID/GID
RUN groupadd -g ${GID} ${USERNAME} \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} \
    && echo "${USERNAME}:${PASSWORD}" | chpasswd \
    && echo "${USERNAME} ALL=(ALL) ALL" >> /etc/sudoers

# Create workspace directories
RUN mkdir -p /workspace/landing \
    && chown -R ${UID}:${GID} /workspace

# Set working directory
WORKDIR /workspace

# Copy requirements first for caching (as root for pip install)
COPY requirements.txt /workspace/requirements.txt

# Install PyTorch with CUDA 12.4 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install numpy first with version constraint (gym 0.23.1 requires numpy < 2.0)
RUN pip install "numpy>=1.24.0,<2.0.0"

# Install remaining Python dependencies
RUN pip install -r requirements.txt

# Copy resources folder to /workspace/resources
# Note: Place the 'resources' folder in the build context before building
COPY resources /workspace/resources

# Fix ownership of workspace after copying files
RUN chown -R ${UID}:${GID} /workspace

# Verify installations
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" \
    && python -c "import numpy; print(f'NumPy version: {numpy.__version__}')" \
    && python -c "import pybullet; print('PyBullet imported successfully')" \
    && python -c "import hydra; print('Hydra imported successfully')" \
    && python -c "from hydra_plugins.hydra_submitit_launcher.submitit_launcher import LocalLauncher; print('Hydra submitit launcher imported successfully')" \
    && python -c "import gym; print(f'Gym version: {gym.__version__}')" \
    && python -c "import shapely; print(f'Shapely version: {shapely.__version__}')" \
    && python -c "import xacro; print('xacro imported successfully')" \
    && python -c "import importlib.metadata as m; print('xacro dist version:', m.version('xacro'))" \
    && command -v xacro >/dev/null && echo "xacro CLI available"

# Switch to non-root user
USER ${USERNAME}

# Default command
CMD ["/bin/bash"]
