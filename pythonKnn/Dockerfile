FROM ubuntu:18.04

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get --yes -qq update \
 && apt-get --yes -qq upgrade 

# mpi 
RUN apt-get --yes -qq install \
                      make \
                      bzip2 \
                      cmake \
                      cpio \
                      curl \
                      g++ \
                      gcc \
                      gfortran \
                      git \
                      gosu \
                      libblas-dev \
                      liblapack-dev \
                      libopenmpi-dev \
                      openmpi-bin \
                      python3-dev \
                      python3-pip \
                      virtualenv \
                      wget \
                      zlib1g-dev \
                      vim       \
                      htop      \
                      software-properties-common \
                      libopenmpi-dev \
 && apt-get --yes -qq clean \
 && rm -rf /var/lib/apt/lists/*

#python
RUN add-apt-repository ppa:deadsnakes/ppa \
 && apt-get --yes -qq install python3.8 python3-pip 


# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip3 install -r requirements.txt

CMD [ "/bin/bash" ]