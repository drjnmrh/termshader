# TERMSHADER

Shading terminal symbols using CUDA and Unicode. Just for lulz.

## Demo

https://github.com/drjnmrh/termshader/assets/18217667/7875be98-1b83-4866-b337-7ad926401877

## How does it work?

I've kept it simple: fire is drawn using stdout and colored space symbol. I've used shader made by Ian McEwan, Ashima Arts (https://www.shadertoy.com/view/MlKSWm).</br>

Shader was rewritten in C++ using CUDA.

# Prerequisites

Currently I've supported only Linux platform (tested on Ubuntu 22.04.3 LTS). Also it is required to launch in a terminal with 24 bit color support.
