# Kernel Image Processing in C++OpenMP/CUDA

Kernel Image Processing is a set of image filtering techniques implemented through a convolution operation between an input image and a mask (kernel).
The kernel is a small matrix that is applied to every pixel of the image: the new value of the pixel is calculated by multiplying the kernel with the surrounding pixels and summing them.

The aim of this work is to compare the computational costs of three different approaches to kernel image processing: sequential C++, parallel OpenMP for CPU, and CUDA for GPU.

## Overview
This project is a final term project for the course of Parallel Computing, held by professor Marco Bertini at University of Florence.

## Requirements
In order to run the code are required:
- OpenMP
- C++17 or newer 
- CUDA
