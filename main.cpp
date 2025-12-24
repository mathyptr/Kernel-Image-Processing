#include <iostream>
#include <string>
#include <chrono>

#include <omp.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "util.h"
#include "imgProc.h"


// Tipi di memoria CUDA 
#define CMD_CUDA_GLOBAL  "global"      // Memoria globale GPU
#define CMD_CUDA_SHARED  "shared"      // Memoria condivisa GPU
#define CMD_CUDA_CONST   "constant"    // Memoria costante GPU


// Tipi di filtro 
#define EDGE_FILTER         "edge"        // Filtro per rilevamento bordi
#define SHARPEN_FILTER      "sharpen"     // Filtro per nitidezza
#define GAUSSIAN_FILTER     "gaussian"    // Filtro per sfocatura gaussiana
#define LAPLACIAN_FILTER    "laplacian"   // Filtro laplaciano


int main(int argc, char** argv) {
    // Verifica i parametri di input

    SplashScreen();
    std::string Filter="gaussian";
    std::string imagePath = "/mnt/datadisk1/c++/clion/Kernel-Image-Processing/input/peperoni.png";
    std::string cudaMemType =  CMD_CUDA_CONST;
    std::string outputdir="/mnt/datadisk1/c++/clion/Kernel-Image-Processing/output/";
    std::string img_ext=".png";
    kernelImgFilter filter;
    if (Filter == GAUSSIAN_FILTER) {
        filter.buildGaussian(7, 1.0f);  // Kernel 7x7, sigma=1.0
    }
    else if (Filter == SHARPEN_FILTER) {
        filter.buildSharp();
    }
    else if (Filter == EDGE_FILTER) {
        filter.buildEdgeDetect();
    }
    else if (Filter == LAPLACIAN_FILTER) {
        filter.buildLaplacian();
    }
    else {
        std::cerr << "Tipo di filtro non valido: " << Filter << std::endl;
        return 1;
    }

    // Visualizza il kernel del filtro selezionato
    std::cout << "\nFiltro da applicare: " << Filter << std::endl;
    filter.display();

    /**
     * Load immagine
     */
    ImgProc inputImage;
    if (!inputImage.loadImageFromFile(imagePath.c_str())) {
        std::cerr << "Errore nella lettura dell'immagine sorgente: " << imagePath << std::endl;
        return 1;
    }

    ImgProc outputCUDA;
    ImgProc outputCPU;
    std::vector<ImgProc> outputsOpenMP;


    /*** Test Elaborazione CUDA ***/
    std::cout << "#############################" << std::endl;
    std::cout << "###Test Elaborazione CUDA ###" << std::endl;
    std::cout << "###  Versione Parallela   ###" << std::endl;
    std::cout << "#############################" << std::endl;
    // Determina il tipo di memoria CUDA da utilizzare
    CudaMemoryType memType = CudaMemoryType::CONSTANT_MEM;
    if (cudaMemType == CMD_CUDA_GLOBAL) {
        memType = CudaMemoryType::GLOBAL_MEM;
    }
    else if (cudaMemType == CMD_CUDA_SHARED) {
        memType = CudaMemoryType::SHARED_MEM;
    }

    
    cudaFree(0);  // Inizializza il runtime CUDA


    auto t3 = std::chrono::high_resolution_clock::now();
    bool cudaResult = inputImage.ParallelFilter(outputCUDA, filter, memType);
    auto t4 = std::chrono::high_resolution_clock::now();

    if (cudaResult) {
        auto cudaDuration = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
        std::cout << "Tempo di esecuzione CUDA: " << cudaDuration << " microsec" << std::endl;
        std::string outputCudaPath = outputdir + std::string("cuda_") + Filter + img_ext;
        outputCUDA.saveImageToFile(outputCudaPath.c_str());
    }

    return 0;
}
