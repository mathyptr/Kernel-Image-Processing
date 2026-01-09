// image_processor.cu
#include "imgProcCuda.h"
#include <png++/png.hpp>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define BLOCK_SIZE BLOCK_DIM_X

#define MAX_KERNEL_SIZE 25


__device__ __constant__ float d_filterKernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];


unsigned int numBlocchi(unsigned int total, unsigned int blockSize) {
    return ceil(total/ blockSize);
}


__global__ void processGlobalMem(
    float* padded, float* d_output, float* kernel,
    int width, int height, int paddedWidth, int paddedHeight,
    int kernelSize)
{

    const int x = blockIdx.x * blockDim.x + threadIdx.x;  // Coordinata x del pixel
    const int y = blockIdx.y * blockDim.y + threadIdx.y;  // Coordinata y del pixel

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int ky = 0; ky < kernelSize; ky++) {
        int py = (y + ky)* paddedWidth +x;
        int pky=ky * kernelSize;
        for (int kx = 0; kx < kernelSize; kx++) {
            float pixelValue = padded[py + kx];
            float kernelValue = kernel[pky + kx];
            sum += pixelValue * kernelValue;
        }
    }

    sum = fmaxf(0.0f, fminf(sum, 255.0f));
    d_output[y * width + x] = sum;
}

__global__ void processConstantMem(
    float* padded, float* d_output,
    int width, int height, int paddedWidth, int paddedHeight,
    int kernelSize)
{

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int ky = 0; ky < kernelSize; ky++) {
        int py = (y + ky)* paddedWidth +x;
        int pky=ky * kernelSize;
        for (int kx = 0; kx < kernelSize; kx++) {
            float pixelValue = padded[py + kx];
            float kernelValue = d_filterKernel[pky + kx];
            sum += pixelValue * kernelValue;
        }
    }

    sum = fmaxf(0.0f, fminf(sum, 255.0f));
    d_output[y * width + x] = sum;
}


__global__ void processSharedMem(
        float* d_input, float* d_output,
        int width, int height, int paddedWidth, int paddedHeight,
        int kernelSize)
{
    const int radius = kernelSize / 2;

    __shared__ float sharedMem[BLOCK_SIZE+MAX_KERNEL_SIZE/2][BLOCK_SIZE+MAX_KERNEL_SIZE/2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int x = blockIdx.x * (blockDim.x-2*radius) + tx;
    const int y = blockIdx.y * (blockDim.y-2*radius) + ty;

    const int xx = blockIdx.x * (blockDim.x-2*radius) + tx-radius;
    const int yy = blockIdx.y * (blockDim.y-2*radius) + ty-radius;

    if (y >= 0 && y < paddedHeight && x >= 0 && x < paddedWidth)
           sharedMem[ty][tx] =  d_input[y * paddedWidth + x];


    __syncthreads();

    if (x>=0 && x < (width+radius) && y>=0 && y < (height+radius))
    {
      if (tx>=radius && tx < (blockDim.x-radius) && ty>=radius && ty < (blockDim.y-radius))
      {
        float sum = 0.0f;
        for (int ky = 0; ky < kernelSize; ky++) {
            for (int kx = 0; kx < kernelSize; kx++) {
                float pixelValue = sharedMem[ky+ty-radius][kx+tx-radius];
                float kernelValue = d_filterKernel[ky * kernelSize +kx ];
                sum += pixelValue * kernelValue;
            }
        }
        sum = fmaxf(0.0f, fminf(sum, 255.0f));
        d_output[yy * width + xx] = sum;
      }
    }

}



ImgProcCuda::ImgProcCuda(ImgProc& inputImage) {
    imgProcInput=inputImage;
    height=imgProcInput.getHeight();
    width=imgProcInput.getWidth();
}


ImgProcCuda::~ImgProcCuda() {

}

bool ImgProcCuda::applyFilter(const kernelImgFilter& filter, const CudaMemoryType memType)
{
    int kernelSize = filter.getSize();
    if (kernelSize > MAX_KERNEL_SIZE) {
        std::cerr << "Dimensione kernel troppo grande" << std::endl;
        return false;
    }

    int radius = kernelSize / 2;
    auto padded = imgProcInput.buildPaddedImg(radius, radius);

    int paddedWidth = width + 2 * radius;
    int paddedHeight = height + 2 * radius;

    dim3 blockSize(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridSize(
            numBlocchi(width, BLOCK_DIM_X),
            numBlocchi(height, BLOCK_DIM_Y)
    );

    dim3 blockSize1(BLOCK_DIM_X+2*radius, BLOCK_DIM_Y+2*radius);

    float* d_input, * d_output, * d_kernel = nullptr;
    if(cudaMalloc(&d_input, paddedWidth * paddedHeight * sizeof(float))!=cudaSuccess)
        return false;
    if(cudaMalloc(&d_output, width * height * sizeof(float))!=cudaSuccess) {
        cudaFree(d_input);
        return false;
    }
    cudaMemcpy(d_input, padded.data(),paddedWidth * paddedHeight * sizeof(float),cudaMemcpyHostToDevice);

    if (memType == CudaMemoryType::GLOBAL_MEM) {
        if(cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float))!=cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            return false;
        }
        cudaMemcpy(d_kernel, filter.getKernelData().data(),kernelSize * kernelSize * sizeof(float),cudaMemcpyHostToDevice);

        processGlobalMem <<<gridSize, blockSize >>> (
            d_input, d_output, d_kernel,
            width, height, paddedWidth, paddedHeight,
            kernelSize
            );
    }
    else if (memType == CudaMemoryType::CONSTANT_MEM) {
        cudaMemcpyToSymbol(d_filterKernel, filter.getKernelData().data(),kernelSize * kernelSize * sizeof(float));

        processConstantMem <<<gridSize, blockSize >>> (
                d_input, d_output,
                width, height, paddedWidth, paddedHeight,
                kernelSize
                );
    }
    else if (memType == CudaMemoryType::SHARED_MEM) {
        cudaMemcpyToSymbol(d_filterKernel, filter.getKernelData().data(),kernelSize * kernelSize * sizeof(float));

//        processSharedMem <<<gridSize, blockSize >>> (
        processSharedMem <<<gridSize, blockSize1 >>> (
                d_input, d_output,
                width, height, paddedWidth, paddedHeight,
                kernelSize
        );

    }
    else
        return false;

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        std::cerr << "Errore CUDA: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    std::vector<float> result(width * height);
    cudaMemcpy(result.data(), d_output,width * height * sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    if (d_kernel) cudaFree(d_kernel);

    imgProc.setImageData(result, width, height);
    return true;
}

bool ImgProcCuda::saveImageToFile(const char* filepath) const {
    try {
        std::vector<float> imgData= imgProc.getImageData();
        png::image<png::gray_pixel> image(width, height);
        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {
                image[y][x] = static_cast<png::gray_pixel>(imgData[y * width + x]);
            }
        }
        image.write(filepath);
        std::cout << "Immagine salvata in: " << filepath << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Errore nel salvataggio dell'immagine: " << e.what() << std::endl;
        return false;
    }
}
