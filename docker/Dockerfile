ARG CUDA="10.2"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

# Install Miniconda
#RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
# && chmod +x /miniconda.sh \
# && miniconda.sh -b -p /miniconda \
# && bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b \
# && rm /miniconda.sh
#RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}




RUN conda update -y conda

#RUN wget \
#    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#    && mkdir /root/.conda \
#    && bash Miniconda3-latest-Linux-x86_64.sh -b \
#    && rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/miniconda/bin:$PATH
#ENV PATH=/root/anaconda/bin:$PATH

#RUN conda --version



# Create a Python 3.6 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name py36 python=3.6 \
 && /miniconda/bin/conda clean -ya

#RUN conda create --name py36 python=3.6

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython
RUN pip install requests ninja yacs cython matplotlib opencv-python tqdm overrides
RUN pip install scipy

# Install PyTorch 1.4
ARG CUDA
RUN conda install pytorch=1.4 torchvision cudatoolkit=${CUDA} -c pytorch \
 && conda clean -ya

# Install TorchVision master
#RUN git clone https://github.com/pytorch/vision.git \
# && cd vision \
# && python setup.py install

RUN export INSTALL_DIR=$PWD

# install pycocotools
RUN cd $INSTALL_DIR

# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install

ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}

# install apex
RUN cd $INSTALL_DIR

#RUN nvidia-smi
ENV TORCH_CUDA_ARCH_LIST="6.0"
RUN git clone https://github.com/NVIDIA/apex.git \
 && cd apex \
 && python setup.py install --cuda_ext --cpp_ext


# install PyTorch Detection
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}
RUN nvidia-smi
RUN git clone https://github.com/gladcolor/Scene-Graph-Benchmark.pytorch.git \
 && cd Scene-Graph-Benchmark.pytorch \
 && python setup.py build develop

RUN unset INSTALL_DIR

