FROM ubuntu:18.04

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

#python
RUN apt-get --yes -qq update \
 && apt-get --yes -qq upgrade \
 && apt-get install --yes -qq software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get --yes -qq install python3.8 python3-pip \
 && pip3 install tqdm

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
 && apt-get --yes -qq clean \
 && rm -rf /var/lib/apt/lists/*

CMD [ "/bin/bash" ]

# set the working directory in the container
#WORKDIR /code

# copy the content of the local src directory to the working directory
#COPY src/ .

#RUN /usr/bin/make

# compile the code
#RUN Make

# copy the content of the local src directory to the working directory
#RUN ls -l

#CMD ["python3", "testAttributi.py"]

#docker run -it --mount src="$(pwd)",target=/home,type=bind openimage
