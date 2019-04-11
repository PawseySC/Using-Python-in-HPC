FROM pawsey/hpc-python

# Install dependencies
RUN apt-get update && \
      apt-get -yq dist-upgrade && \
      apt-get install -yq --no-install-recommends \
        dirmngr \
        libopenblas-dev \
        python3-dev \
        scotch \
        metis && \
      pip install --no-cache-dir \
        appdirs \
        appdirs \
        mako \
        pytools && \
      rm -rf ~/.cache/pip


# Install CUDA
### Install CUDA ###
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn
WORKDIR /tmp/cuda-build
ADD cuda-repo-ubuntu1710_9.2.148-1_amd64.deb .
RUN dpkg -i cuda-repo-ubuntu1710_9.2.148-1_amd64.deb \
      && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1710/x86_64/7fa2af80.pub \
      && apt-get update \
      && apt-get install -yq --no-install-recommends cuda \
      && apt-get clean all \
      && rm -r /var/lib/apt/lists/*


# Build mpi4py
ARG MPI4PY_VERSION="3.0.0"
WORKDIR /tmp
ADD https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-${MPI4PY_VERSION}.tar.gz .
RUN tar xf mpi4py-${MPI4PY_VERSION}.tar.gz && \
      cd mpi4py-${MPI4PY_VERSION} && \
      python3 setup.py build && \
      python3 setup.py install


# Install Gimmik
WORKDIR /tmp/gimmik-build
ADD https://github.com/vincentlab/GiMMiK/archive/v2.1.tar.gz .
RUN tar xf v2.1.tar.gz && \
      cd GiMMiK-2.1 && \
      python3 setup.py install


# Install PyFR
WORKDIR /tmp/pyfr-build
ADD http://www.pyfr.org/download/PyFR-1.8.0.zip && \
RUN unzip PyFR-1.8.0.zip && \
      cd PyFR-1.8.0 && \
      python3 setup.py install


# Clean up
WORKDIR /
RUN rm -rf /tmp
CMD ["/bin/bash"]
