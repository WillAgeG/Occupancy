# took from https://gitlab.sberautotech.ru/nrmaslovich/clearmlperceptiontemplates/-/blob/dev/Dockerfile@agent_pvm?ref_type=heads
FROM registry.sberautotech.ru:5574/sd/dl/base-images/clearml:cuda-12.6.3-cudnn as base

USER root
# Configure apt and install packages
RUN apt-get update \
    # Install tools
    && apt-get install -y apt-utils \
    && apt-get install -y --no-install-recommends lsb-release \
                        iputils-ping cmake cppcheck valgrind git git-lfs pylint \
                        wget curl build-essential python3-pip python3-venv python3-setuptools \
                        python3-apt python3-dev virtualenv sudo micro unzip wget ffmpeg libsm6 libxext6 \
    && git lfs install \                        
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Installing python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

USER $USER_NAME

# Specify paths
ENV PYTHONPATH=$PYTHONPATH:/workspaces/NavioOccupancy/
ENV PATH=/workspaces/NavioOccupancy/:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /workspaces

COPY --chown=$USER_NAME:$GROUP_NAME project2d /workspaces/NavioOccupancy/project2d

# ----- ONLY FOR LOCAL DEVELOPMENT -------------------
FROM base as local

USER root

# Create local user
ARG UID=1000
ARG GID=6000

RUN usermod -u $UID $USER_NAME
RUN groupmod -g $GID $GROUP_NAME

RUN pip install jupyter

USER $USER_NAME