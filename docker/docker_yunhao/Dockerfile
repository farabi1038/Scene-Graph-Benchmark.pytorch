FROM  biggreenapple/scene-graph:latest

RUN echo 'Started...'


RUN pip install scipy
RUN pip install h5py

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libopencv-dev \
        libsnappy-dev \
        python-dev \
        python-pip \
        tzdata \
        vim

# Install anaconda for python 3.6
RUN curl -LO  https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    echo "export PATH=/opt/conda/bin:$PATH" >> ~/.bashrc