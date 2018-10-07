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
Pkg.add("Plots)
Pkg.add("Knet")
Pkg.add("JLD2")
Pkg.add("DSP")
```
on Linux the `Dieckx` package needs a fortran compiler so `sudo apt-get install gfortran`
1. clone this repository `git clone https://github.com/imagry/aleph_star`
2. 

## The environment

The only supplied environment is of a car following a lane. It can be found in the `env` directory. The environment puts a car 