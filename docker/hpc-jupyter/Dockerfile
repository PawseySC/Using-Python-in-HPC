# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

# Beginning of base-notebook

# Ubuntu 18.04 (bionic) from 2018-05-26
# https://github.com/docker-library/official-images/commit/aac6a45b9eb2bffb8102353c350d341a410fb169
ARG BASE_CONTAINER=ubuntu:bionic-20180526@sha256:c8c275751219dadad8fa56b3ac41ca6cb22219ff117ca98fe82b42f24e1ba64e
FROM $BASE_CONTAINER

MAINTAINER Marco De La Pierre <marco.delapierre@pawsey.org.au

ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

USER root

# Install all OS dependencies for notebook server that starts but lacks all
# features (e.g., download as all possible file formats)
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
      apt-get -yq dist-upgrade && \
      apt-get install -yq --no-install-recommends \
      bzip2 \
      build-essential \
      ca-certificates \
      dvipng \
      emacs \
      ffmpeg \
      fonts-liberation \
      gfortran \
      gdb \
      git \
      inkscape \
      jed \
      libopenblas-dev \
      libsm6 \
      libxext-dev \
      libxrender1 \
      lmodern \
      locales \
      nano \
      netcat \
      pandoc \
      python-dev \
      sudo \
      texlive-fonts-extra \
      texlive-fonts-recommended \
      texlive-generic-recommended \
      texlive-latex-base \
      texlive-latex-extra \
      texlive-xetex \
      tzdata \
      unzip \
      valgrind \
      vim \
      wget && \
      apt-get clean all && \
      rm -r /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
      locale-gen

# Configure environment
ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    NB_USER=$NB_USER \
    NB_UID=$NB_UID \
    NB_GID=$NB_GID \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8
ENV PATH=$CONDA_DIR/bin:$PATH \
    HOME=/home/$NB_USER

# Add a script that we will use to correct permissions after running certain commands
ADD fix-permissions /usr/local/bin/fix-permissions

# Enable prompt color in the skeleton .bashrc before creating the default NB_USER
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc

# Create NB_USER wtih name jovyan user with UID=1000 and in the 'users' group
# and make sure these dirs are writable by the `users` group.
RUN groupadd wheel -g 11 && \
      echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
      useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
      mkdir -p $CONDA_DIR && \
      chown $NB_USER:$NB_GID $CONDA_DIR && \
      chmod g+w /etc/passwd && \
      fix-permissions $HOME && \
      fix-permissions "$(dirname $CONDA_DIR)"

# Setup work directory for backward-compatibility
USER $NB_UID
RUN mkdir /home/$NB_USER/work && \
      fix-permissions /home/$NB_USER

# Install conda as jovyan and check the md5 sum provided on the download site
ENV MINICONDA_VERSION=4.5.12 \
  CONDA_VERSION=4.6.7

RUN cd /tmp && \
      wget --quiet https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
      echo "866ae9dff53ad0874e1d1a60b1ad1ef8 *Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" | md5sum -c - && \
      /bin/bash Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
      rm Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
      $CONDA_DIR/bin/conda config --system --prepend channels conda-forge && \
      $CONDA_DIR/bin/conda config --system --set auto_update_conda false && \
      $CONDA_DIR/bin/conda config --system --set show_channel_urls true && \
      $CONDA_DIR/bin/conda install --quiet --yes conda="${CONDA_VERSION%.*}.*" && \
      $CONDA_DIR/bin/conda update --all --quiet --yes


# Install Jupyter Notebook, Lab, Hub, and Python 3 packages
# Install notebook extensions
# Generate a notebook server config
# Cleanup temporary files and remove pyqt/qt (only using notebook compatible backends for matplotlib)
# Correct permissions
# Do all this in a single RUN command to avoid duplicating all of the
# files across image layers when the permissions change
RUN conda install --quiet --yes \
    'astropy' \
    'beautifulsoup4=4.7.*' \
    'bokeh=1.0*' \
    'cloudpickle=0.8*' \
    'conda-forge::blas=*=openblas' \
    'cython=0.29*' \
    'dask=1.1.*' \
    'dill=0.2*' \
    'h5py=2.9*' \
    'hdf5=1.10*' \
    'ipyparallel' \
    'ipywidgets=7.4*' \
    'jupyter_nbextensions_configurator' \
    'jupyterhub=0.9.6' \
    'jupyterlab=0.35.4' \
    'joblib' \
    'matplotlib=3.0*' \
    'nb_conda' \
    'netCDF4' \
    'notebook=5.7.8' \
    'numba=0.42*' \
    'numexpr=2.6*' \
    'pandas=0.24*' \
    'patsy=0.5*' \
    'prettytable' \
    'protobuf=3.7.*' \
    'pytables' \
    'rise' \
    'scikit-image=0.14*' \
    'scikit-learn=0.20*' \
    'scipy=1.2*' \
    'seaborn=0.9*' \
    'sqlalchemy=1.3*' \
    'statsmodels=0.9*' \
    'sympy=1.3*' \
    'tini=0.18.0' \
    'vincent=0.4.*' \
    'xlrd' && \
    conda remove --quiet --yes --force qt pyqt && \
    conda list tini | grep tini | tr -s ' ' | cut -d ' ' -f 1,2 >> $CONDA_DIR/conda-meta/pinned && \
    conda clean -tipsy && \
    jupyter labextension install @jupyterlab/hub-extension@^0.12.0 && \
    jupyter nbextension enable --py widgetsnbextension --sys-prefix && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager@^0.38.1 && \
    jupyter labextension install jupyterlab_bokeh@0.6.3 && \
    ipcluster nbextension enable --user && \
    jupyter nbextensions_configurator enable --user && \
    pip install jupyter_contrib_nbextensions && \
    jupyter contrib nbextension install --user && \
    jupyter nbextension enable spellchecker/main --user && \
    npm cache clean --force && \
    jupyter notebook --generate-config && \
    rm -rf $CONDA_DIR/share/jupyter/lab/staging && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    rm -rf /home/$NB_USER/.node-gyp

# Install facets which does not have a pip or conda package at the moment
RUN cd /tmp && \
      git clone https://github.com/PAIR-code/facets.git && \
      cd facets && \
      jupyter nbextension install facets-dist/ --sys-prefix && \
      fix-permissions $CONDA_DIR && \
      fix-permissions /home/$NB_USER

# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME /home/$NB_USER/.cache/
RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot" && \
    fix-permissions /home/$NB_USER

# Build PyFR
RUN cd /tmp && \
      wget http://www.pyfr.org/download/PyFR-1.8.0.zip && \
      wget https://github.com/vincentlab/GiMMiK/archive/v2.1.tar.gz && \
      unzip PyFR-1.8.0.zip && \
      tar xf v2.1.tar.gz && \
      conda create -y -n pyfr \
      appdirs \
      h5py \
      ipyparallel \
      mako \
      metis \
      mpi4py \
      nb_conda \
      numpy \
      pytools \
      scotch
# Create our conda env for pyfr
RUN cd /tmp/GiMMiK-2.1 && \
    /bin/bash -c ". /opt/conda/etc/profile.d/conda.sh && \
    conda activate pyfr && \
    python setup.py install && \
    cd /tmp/PyFR-1.8.0 && \
    python setup.py install && \
    conda deactivate"

# Build MPICH
USER root
ARG MPICH_VERSION="3.1.4"
ARG MPICH_CONFIGURE_OPTIONS="--enable-fast=all,O3 --prefix=/usr"
ARG MPICH_MAKE_OPTIONS="-j4"
WORKDIR /tmp/mpich-build
ADD http://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz .
RUN tar xf mpich-${MPICH_VERSION}.tar.gz && \
      cd mpich-${MPICH_VERSION} && \
      ./configure ${MPICH_CONFIGURE_OPTIONS} && \
      make ${MPICH_MAKE_OPTIONS} && \
      make install && \
      ldconfig

# Build mpi4py
ARG MPI4PY_VERSION="3.0.0"
WORKDIR /tmp
ADD https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-${MPI4PY_VERSION}.tar.gz .
RUN tar xf mpi4py-${MPI4PY_VERSION}.tar.gz && \
      cd mpi4py-${MPI4PY_VERSION} && \
      python setup.py build && \
      python setup.py install

# Build PETSc
ARG PETSC_VERSION="3.10.5"
ARG PETSC_CONFIGURE_OPTIONS="--prefix=/usr --with-openblas=1"
WORKDIR /tmp/petsc-build
ADD http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-${PETSC_VERSION}.tar.gz .
RUN tar xf petsc-lite-${PETSC_VERSION}.tar.gz && \
      cd petsc-${PETSC_VERSION} && \
      ./configure ${PETSC_CONFIGURE_OPTIONS} && \
      make && \
      make install

# Build petsc4py
ARG PETSC4PY_VERSION="3.10.1"
WORKDIR /tmp/petsc4py-build
ENV PETSC_DIR=/usr
ADD https://bitbucket.org/petsc/petsc4py/downloads/petsc4py-${PETSC4PY_VERSION}.tar.gz .
RUN ls && tar xf petsc4py-${PETSC4PY_VERSION}.tar.gz && \
      cd petsc4py-${PETSC4PY_VERSION} && \
      python setup.py build && \
      python setup.py install

# Clean up
USER root
WORKDIR /
RUN fix-permissions $CONDA_DIR && \
      fix-permissions /home/$NB_USER && \
      rm -rf /tmp/*

# Add local files as late as possible to avoid cache busting
COPY start.sh /usr/local/bin/
COPY start-notebook.sh /usr/local/bin/
COPY start-singleuser.sh /usr/local/bin/
COPY jupyter_notebook_config.py /etc/jupyter/
RUN fix-permissions /etc/jupyter/

# Configure container startup
EXPOSE 8888
WORKDIR $HOME
ENTRYPOINT ["tini", "-g", "--"]
CMD ["start-notebook.sh"]

USER $NB_UID
