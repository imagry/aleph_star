<div align="center">
  <img src="./aleph_star_logo.png">
</div>

This repository contains the necessary code to reproduce the results presented in the [GTC 2018 (Israel)](https://www.nvidia.com/en-il/gtc/) talk and WIP paper "Reinforcement learning with A* and a deep heuristic". We call this algorithm Aleph-Star.

The code is not yet in a package format (this is WIP), however it is fairly general and can easily be used with different environments.

## The environment

The only supplied environment is of a car following a lane. It can be found in the `env` directory. The environment puts a car 