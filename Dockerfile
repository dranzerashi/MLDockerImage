FROM nvidia/cuda:9.0-cudnn7-runtime
MAINTAINER Infosys STGAT Team "akshatha_holla@infosys.com"

RUN apt-get update && apt-get install -y \
    apt-transport-https \
    iputils-ping \
    git \
    software-properties-common \
    build-essential \
    cmake \
    libhdf5-dev \
    swig \
    wget \
    curl

## Python 3.6
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y  && \
    apt-get install python3.6 -y \
        python3.6-venv \
        python3.6-dev \
        python3-software-properties

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6

### Without this Python thinks we're ASCII and unicode chars fail
ENV LANG C.UTF-8
## Set up python3.6 environment

RUN pip3.6 install -U pip
RUN pip3.6 install -U \
      numpy \
      scipy \
      matplotlib \
      pandas \
      sympy \
      nose \
      tqdm \
      wheel \
      scikit-learn \
      scikit-image \
      dlib 

RUN pip3.6 install -U \
      tensorflow-gpu \
      tensorboard \
      keras

# Workspace creation
ADD . /code
WORKDIR /code
#CMD ["python3.6", "app.py"]