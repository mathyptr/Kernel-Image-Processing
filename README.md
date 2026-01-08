# Kernel Image Processing in C++ OpenMP/CUDA

## Description
Kernel Image Processing is a project focused on digital image filtering using **convolution kernels**. A kernel is a small matrix applied to every pixel of an image in order to compute a new value based on the surrounding pixels. This technique is widely used for image processing tasks such as smoothing, sharpening, and edge detection.

The project provides three different implementations of kernel-based image processing:
- **Sequential C++**
- **Parallel CPU implementation using OpenMP**
- **Parallel GPU implementation using CUDA**

The purpose is to compare their computational performance and analyze the benefits of parallelism on different architectures.

---

## Objectives
- Implement kernel-based image filtering using convolution
- Compare sequential and parallel approaches
- Evaluate performance differences between CPU and GPU execution
- Study the scalability and efficiency of OpenMP and CUDA

---

## Technologies Used
- **C++ (C++17 or newer)**
- **OpenMP** for CPU parallelization
- **CUDA** for GPU acceleration

---

## Requirements
To compile and run the project, the following are required:
- A C++ compiler supporting **C++17 or newer**
- **OpenMP** support
- **NVIDIA CUDA Toolkit**
- A CUDA-capable GPU (for GPU execution)

---

## Project Context
This project was developed as a **final term project** for the course **Parallel Computing**, held by **Professor Marco Bertini** at the **University of Florence**.

---

## Notes
- Performance results may vary depending on hardware configuration.
- CUDA implementation requires a compatible NVIDIA GPU.
