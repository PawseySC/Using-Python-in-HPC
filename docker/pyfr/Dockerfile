FROM ubuntu:18.04

# Install dependencies
RUN apt-get update && \
      apt-get -yq dist-upgrade && \
      apt-get install -yq --no-install-recommends \
        build-essential \
        dirmngr \
        gdb \
        gfortran \
        gnupg \
        libmetis-dev \
        libopenblas-dev \
        python3-appdirs \
        python3-dev \
        python3-mako \
        python3-pip \
        python3-pytools \
        python3-setuptools \
        unzip \
        wget && \
      apt-get clean all && \
      rm -r /var/lib/apt/lists/*


### Build MPICH ###
ARG MPICH_VERSION="3.1.4"
ARG MPICH_CONFIGURE_OPTIONS="--enable-fast=all,O3 --prefix=/usr"
ARG MPICH_MAKE_OPTIONS="-j4"
WORKDIR /tmp/mpich-build
RUN wget http://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz \
      && tar xvzf mpich-${MPICH_VERSION}.tar.gz \
      && cd mpich-${MPICH_VERSION}  \
      && ./configure ${MPICH_CONFIGURE_OPTIONS} \
      && make ${MPICH_MAKE_OPTIONS} \
      && make install \
      && ldconfig


### Install CUDA ###
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn
WORKDIR /tmp/cuda-build
ADD cuda-repo-ubuntu1804_10.1.105-1_amd64.deb .
RUN dpkg -i cuda-repo-ubuntu1804_10.1.105-1_amd64.deb \
      && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1710/x86_64/7fa2af80.pub \
      && apt-get update \
      && apt-get install -yq --no-install-recommends cuda \
      && apt-get clean all \
      && rm -r /var/lib/apt/lists/*


### Build mpi4py ###
ARG MPI4PY_VERSION="3.0.0"
WORKDIR /tmp
ADD https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-${MPI4PY_VERSION}.tar.gz .
RUN tar xf mpi4py-${MPI4PY_VERSION}.tar.gz && \
      cd mpi4py-${MPI4PY_VERSION} && \
      python3 setup.py build && \
      python3 setup.py install


### Install Gimmik ###
WORKDIR /tmp/gimmik-build
ADD https://github.com/vincentlab/GiMMiK/archive/v2.1.tar.gz .
RUN tar xf v2.1.tar.gz && \
      cd GiMMiK-2.1 && \
      python3 setup.py install


### Install PyCUDA
ENV PATH=/usr/local/cuda/bin:$PATH
WORKDIR /tmp/pycuda-build
ADD https://files.pythonhosted.org/packages/58/33/cced4891eddd1a3ac561ff99081019fddc7838a07cace272c941e3c2f915/pycuda-2018.1.1.tar.gz .
RUN tar xf pycuda-2018.1.1.tar.gz && \
      cd pycuda-2018.1.1 && \
      python3 configure.py --cuda-root=/usr/local/cuda && \
      make install


### Install PyFR ###
WORKDIR /tmp/pyfr-build
RUN wget http://www.pyfr.org/download/PyFR-1.8.0.zip && \
      unzip PyFR-1.8.0.zip && \
      cd PyFR-1.8.0 && \
      python3 setup.py install


### Clean up ###
WORKDIR /
RUN rm -rf /tmp/*
CMD ["/bin/bash"]
