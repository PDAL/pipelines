## Start with mambaforge and create environment from conda lock file.
FROM condaforge/mambaforge:4.9.2-5 as conda
COPY conda-linux-64.lock .
RUN --mount=type=cache,target=/opt/conda/pkgs mamba create --copy -p /env --file conda-linux-64.lock

## Stick with mambaforge and build Open3D in the conda environment (copied from previous stage).
FROM condaforge/mambaforge:4.9.2-5 AS builder
ENV TZ=America/Chicago
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && \
    apt-get install -y --fix-missing --no-install-recommends \
        autoconf \
        build-essential \
        git \
        gpg \
        libc++-7-dev \
        libc++abi-7-dev \
        libglu1-mesa-dev \
        libosmesa6-dev \
        libsdl2-dev \
        libtbb-dev \
        libtool \
        libudev-dev \
        libxi-dev \
        python3-dev \
        python3-setuptools \
        python3-pip \
        xorg-dev && \
    rm -rf /var/lib/apt/lists/*
SHELL ["/bin/bash", "-c"]
COPY --from=conda /env /env
RUN conda run -p /env pip3 install pdal-plugins
RUN git clone https://github.com/isl-org/Open3D && \
    cd Open3D && mkdir build && cd build && \
    conda run -p /env cmake -DBUILD_EXAMPLES=OFF .. && \
    conda run -p /env make -j8 && \
    conda run -p /env make python-package && \
    conda run -p /env make install-python-package
RUN conda uninstall -p /env cmake pip setuptools && \
    conda clean -afy
RUN find -name '*.a' -delete && \
    rm -rf /env/include && \
    find -name '__pycache__' -type d -exec rm -rf '{}' '+' && \
    rm -rf /env/lib/python3.10/site-packages/pip \
        /env/lib/python3.10/idlelib \
        /env/lib/python3.10/ensurepip \
        /env/lib/libasan.so.5.0.0 \
        /env/lib/libtsan.so.0.0.0 \
        /env/lib/liblsan.so.0.0.0 \
        /env/lib/libubsan.so.1.0.0 \
        /env/bin/x86_64-conda-linux-gnu-ld \
        /env/bin/sqlite3 \
        /env/bin/openssl \
        /env/share/terminfo && \
    find /env/lib/python3.10/site-packages/numpy -name 'tests' -type d -exec rm -rf '{}' '+' && \
    find /env/lib/python3.10/site-packages -name '*.pyx' -delete && \
    rm -rf /env/lib/python3.10/site-packages/uvloop/loop.c

## Use miniconda3 as our final stage, which allows us to use conda run when specifying the entrypoint.
FROM continuumio/miniconda3
RUN apt-get update && \
    apt-get install -y --fix-missing --no-install-recommends \
        libglu1-mesa-dev && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /env /env
COPY script.py .
COPY evaluate.py .
COPY evaluate_precision.py .
COPY evaluate_planetary.py .
COPY registration.py .
WORKDIR /data
CMD ["moving.laz", "fixed.laz", "aligned.laz"]
ENTRYPOINT ["conda", "run", "-p", "/env", "python", "/script.py"]
