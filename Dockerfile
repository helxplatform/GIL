FROM nvidia/cuda:10.2-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    vim \
    make \
    gcc \
    wget \
    tar \
    unzip \
    git
        
RUN apt-get update && apt-get install -y software-properties-common 
RUN add-apt-repository ppa:graphics-drivers/ppa
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.8 python3-pip 
RUN apt-get update && apt-get install -y python3-venv python3-dev 

RUN python3 -m pip install --upgrade pip && python3 -m pip install tensorflow
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

#RUN apt-get update && apt-get install -y nvidia-smi
RUN python3 -m pip install --upgrade pip && python3 -m pip install pandas numpy sklearn 
RUN python3 -m pip install --upgrade pip && python3 -m pip install matplotlib seaborn 
RUN python3 -m pip install --upgrade pip && python3 -m pip install SimpleITK

WORKDIR /opt
RUN git clone https://github.com/helxplatform/GIL.git

COPY Dockerfile /opt
