# Overview

Provides PDAL Python filter that exposes Open3D registration methods.

This example contains the following files:

- Dockerfile: to recreate the Docker image used to package Open3D and PDAL in a compatible manner
- registration.py: the PDAL Python filter that wraps the three available Open3D registration methods: point-to-point ICP, point-to-plane ICP, and colored ICP (in this case, the colors correspond to classification labels)
- script.py: a Python script for evaluating registration.py, e.g., with randomly generated transformation matrices
- pipeline.json: an example PDAL pipeline calling the registration.py source

Future could be useful to have several options:

1. Run the existing script to register two files, just pass the paths
2. Run a PDAL pipeline, passing the fixed filename to pdalargs is messy
3. Run an evaluation script that can iterate over multiple perturbations
4. Run a benchmark script, which would be interesting to also highly specialized

# Installation

Build or pull Docker image.

```bash
$ conda lock -p linux-64 -f environment.yml
$ docker build -t open3d-pdal .
```

```bash
$ docker pull chambbj/open3d-pdal:0.1
```

# Running

The default entrypoint will run a Python script that aligns the provided `moving.laz` with `fixed.laz` producing `aligned.laz`.

```bash
$ docker run -it --rm -v $(pwd):/data open3d-pdal moving.laz fixed.laz aligned.laz
```

We can specify a different entrypoint and set of parameters to run the evaluation script.

```bash
$ docker run -it --rm -v $(pwd):/data --entrypoint conda chambbj/pdal-open3d:0.4 run -p /env python evaluate.py
```

# Pipeline

Just edit the pipeline, providing paths for fixed, moving, and aligned point clouds, then execute with `pdal pipeline pipeline.json`.
