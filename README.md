# TERMSHADER

Shading terminal symbols using CUDA and Unicode. Just for lulz.

## Demo

<video width="320" height="240" controls>
  <source src="UnicodeOnFire.mp4" type="video/mp4">
</video>

## How does it work?

I've kept it simple: fire is drawn using stdout and colored space symbol. Used shader made by Ian McEwan, Ashima Arts (https://www.shadertoy.com/view/MlKSWm).</br>

Shader was rewritten in C++ using CUDA.

# Prerequisites

Currently I've supported only Linux platform (tested on Ubuntu 22.04.3 LTS). Also it is required to have a terminal which supports 24 bit color.
