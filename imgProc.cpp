// image_processor.cu
#include "imgProc.h"
#include <png++/png.hpp>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16


#define MAX_KERNEL_SIZE 25



ImgProc::ImgProc() : width(0), height(0) {}


ImgProc::~ImgProc() {
    imgData.clear();
    std::vector<float>().swap(imgData);
}


bool ImgProc::loadImageFromFile(const char* filepath) {
    try {
        png::image<png::gray_pixel> image(filepath);
        width = image.get_width();
        height = image.get_height();

        imgData.resize(width * height);
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                imgData[y * width + x] = static_cast<float>(image[y][x]);
            }
        }
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Errore nel caricamento dell'immagine: " << e.what() << std::endl;
        return false;
    }
}


bool ImgProc::saveImageToFile(const char* filepath) const {
    try {
        png::image<png::gray_pixel> image(width, height);
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
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


std::vector<float> ImgProc::createPaddedImg(int paddingY, int paddingX) const {
    int paddedWidth = width + 2 * paddingX;
    int paddedHeight = height + 2 * paddingY;
    std::vector<float> padded(paddedWidth * paddedHeight);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            padded[(y + paddingY) * paddedWidth + (x + paddingX)] =
                    imgData[y * width + x];
        }
    }

    for (int y = 0; y < paddingY; ++y) {
        std::copy_n(&padded[paddingY * paddedWidth], width,
            &padded[y * paddedWidth + paddingX]);
        std::copy_n(&padded[(height + paddingY - 1) * paddedWidth], width,
            &padded[(paddedHeight - y - 1) * paddedWidth + paddingX]);
    }


    for (int y = 0; y < paddedHeight; ++y) {
        for (int x = 0; x < paddingX; ++x) {
            padded[y * paddedWidth + x] = padded[y * paddedWidth + paddingX];
            padded[y * paddedWidth + paddedWidth - 1 - x] =
                padded[y * paddedWidth + paddedWidth - paddingX - 1];
        }
    }

    return padded;
}

/*
bool ImgProc::ParallelFilter(
        ImgProc& output, const kernelImgFilter& filter, const CudaMemoryType memType)
{
    int kernelSize = filter.getSize();
    if (kernelSize > MAX_KERNEL_SIZE) {
        std::cerr << "Dimensione kernel troppo grande" << std::endl;
        return false;
    }

    int radius = kernelSize / 2;
    auto padded = createPaddedImg(radius, radius);
    int paddedWidth = width + 2 * radius;
    int paddedHeight = height + 2 * radius;

    float* d_input, * d_output, * d_kernel = nullptr;
    cudaMalloc(&d_input, paddedWidth * paddedHeight * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));

    cudaMemcpy(d_input, padded.data(),
        paddedWidth * paddedHeight * sizeof(float),
        cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridSize(
        calcolaBlocchi(width, BLOCK_DIM_X),
        calcolaBlocchi(height, BLOCK_DIM_Y)
    );

    if (memType == CudaMemoryType::GLOBAL_MEM) {
        cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));
        cudaMemcpy(d_kernel, filter.getKernelData().data(),
            kernelSize * kernelSize * sizeof(float),
            cudaMemcpyHostToDevice);

        processaImmagineGlobale << <gridSize, blockSize >> > (
            d_input, d_output, d_kernel,
            width, height, paddedWidth, paddedHeight,
            kernelSize
            );
    }
    else if (memType == CudaMemoryType::SHARED_MEM) {
        cudaMemcpyToSymbol(d_filterKernel, filter.getKernelData().data(),
            kernelSize * kernelSize * sizeof(float));

        processaImmagineShared<BLOCK_DIM_X> << <gridSize, blockSize >> > (
            d_input, d_output,
            width, height, paddedWidth, paddedHeight,
            kernelSize
            );
    }
    else {  // CONSTANT_MEM
        cudaMemcpyToSymbol(d_filterKernel, filter.getKernelData().data(),
            kernelSize * kernelSize * sizeof(float));

        processaImmagineConstante << <gridSize, blockSize >> > (
            d_input, d_output,
            width, height, paddedWidth, paddedHeight,
            kernelSize
            );
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Errore CUDA: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    std::vector<float> result(width * height);
    cudaMemcpy(result.data(), d_output,
        width * height * sizeof(float),
        cudaMemcpyDeviceToHost);


    cudaFree(d_input);
    cudaFree(d_output);
    if (d_kernel) cudaFree(d_kernel);

    output.setImageData(result, width, height);
    return true;
}
*/

int ImgProc::getWidth() const { return width; }
int ImgProc::getHeight() const { return height; }
int ImgProc::getChannels() const { return 1; }  // Immagini in scala di grigi


bool ImgProc::setImageData(const std::vector<float>& data, int w, int h) {
    if (data.size() != w * h) {
        std::cerr << "Dimensioni dati non valide" << std::endl;
        return false;
    }
    width = w;
    height = h;
    imgData = data;
    return true;
}

std::vector<float> ImgProc::getImageData() const {
    return imgData;
}