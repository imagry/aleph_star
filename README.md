<div align="center">
  <img src="./aleph_star_logo.png">
</div>

This repository contains the necessary code to reproduce the results presented in the [GTC 2018 (Israel)](https://www.nvidia.com/en-il/gtc/) talk and WIP paper "Reinforcement learning with A* and a deep heuristic". We call this algorithm Aleph-Star.

The code is not yet in easily-installable and registered Julia package format (this is WIP), however it is fairly general and can easily be used with different environments.

## Installation

1. Make sure you have [Julia 1.01](https://julialang.org/downloads/) or above installed
2. Install several needed Julia packages:
```Julia
using Pkg
Pkg.add("DataStructures")
Pkg.add("Nullables")
Pkg.add("Dierckx")
Pkg.add("PyPlot")
Pkg.add("IJulia")
Pkg.add("Plots")
Pkg.add("Knet")
Pkg.add("JLD2")
Pkg.add("DSP")
```
on Linux the `Dierckx` package needs a fortran compiler so `sudo apt-get install gfortran`. The ML package is [Knet](https://github.com/denizyuret/Knet.jl) and requires the `nvcc` (NVIDIA compiler from `Cuda`) to be in the path, otherwise it will default to using the CPU, and the code using it has to be modified accordingly. The `IJulia` package is only needed to use the Jupyter notebooks (`jupyter notebook` has to work from the command line). The basic algorithm implementation (in the `aleph_zero` folder) uses only the `DataStructures` and `Nullables` packages. `Dierckx`, `Knet` and `DSP` are used by the environment (described below), `PyPlot` and `Plots` are only used for interactive plotting of results, and `JLD2` is for saving the results in `HDF5` format. `IJulia` is generally not needed, except the Jupyter notebooks are very nice for this kind of work. When This repository becomes a package it will be much easier to install.

3. clone this repository `git clone https://github.com/imagry/aleph_star`
4. You're all set :) load the jupyter notebooks and have fun!

## The environment

The only supplied environment is of a car following a lane. It can be found in the `env` directory, to include it call `include("env/env_lane.jl")`. This environment generates a lane of random width and curves. The curvature, min/max width and length can be configured, but to use the defaults just call `state, env = initialize_simple_road()` which will generate a new random lane and an initial state. From any state `sensors` can be generated, in our case an image: `sensors = get_sensors(env, state)` which can be plotted by (for e.g.) `heatmap(sensors, aspect_ratio=1.0)` resulting in:





