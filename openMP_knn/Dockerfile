FROM ubuntu:18.04

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

#python
RUN apt-get --yes -qq update \
 && apt-get --yes -qq upgrade \
 && apt-get install --yes -qq software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get --yes -qq install python3.8 python3-dev python3-pip \
 && pip3 install tqdm

RUN apt-get --yes -qq upgrade \
 && apt-get --yes -qq install \
                      bzip2 \
                      cmake \
                      make \
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
                      wget \
                      zlib1g-dev \
                      vim       \
                      htop      \
 && apt-get --yes -qq clean \
 && rm -rf /var/lib/apt/lists/*

CMD [ "/bin/bash" ]