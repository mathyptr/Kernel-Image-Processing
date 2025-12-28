#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>

#include <omp.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "util.h"
#include "imgProc.h"
#include "imgProcSeq.h"
#include "imgProcCuda.h"
#include "imgProcOMP.h"



int main(int argc, char** argv) {
    // Verifica i parametri di input

    SplashScreen();
    std::string Filter="gaussian";
    std::string imagePath = "./input/peperoni1024.png";
    std::string outputdir="./output/";
    std::string file_outPar = "./output/resultPAR.csv";
    std::string img_ext=".png";
    kernelImgFilter kImgFilter;

    std::cout << "Directory corrente: " << filesystem::current_path() << '\n';

    std::string  filter;
    filter=chooseFilter();
    kImgFilter.buildFilter(filter);

    // Visualizza il kernel del filtro selezionato
    std::cout << "\nFiltro da applicare: " << kImgFilter.getName() << std::endl;
    kImgFilter.display();

    /**
     * Load immagine
     */
    ImgProc inputImage;
    if (!inputImage.loadImageFromFile(imagePath.c_str())) {
        std::cerr << "Errore nella lettura dell'immagine sorgente: " << imagePath << std::endl;
        return 1;
    }

    testResult testr;
    std::vector<testResult> testVectResult;

    /*** Test Elaborazione Seguenziale ***/
    std::cout << "############################" << std::endl;
    std::cout << "###  Test Elaborazione   ###" << std::endl;
    std::cout << "### Versione SEQUENZIALE ###" << std::endl;
    std::cout << "############################" << std::endl;

    ImgProcSeq imgSeq(inputImage);
    std::chrono::high_resolution_clock::time_point tstart;
    std::chrono::high_resolution_clock::time_point tend;
    bool cpuResult;
    tstart = std::chrono::high_resolution_clock::now();
    cpuResult = imgSeq.applyFilter(kImgFilter);
    tend = std::chrono::high_resolution_clock::now();

    if (cpuResult) {
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
        std::cout << "Tempo di esecuzione CPU: " << elapsed << " microsec" << std::endl;
        std::string outputFileImgaePath = outputdir + std::string("cpu_") + kImgFilter.getName()+ img_ext;
        imgSeq.saveImageToFile(outputFileImgaePath.c_str());

        std::vector<testResult> testVectResultSEQ;
        testr.execTimes=elapsed;
        testr.num_iter= 1;
        testr.test_type=SEQUENTIAL;
        testVectResultSEQ.push_back(testr);
        testVectResult.push_back(testr);
        std::string title="Risultati test Sequenziali";
        SplashResult(title,testVectResultSEQ);
    }




    std::vector<ImgProc> outputsOpenMP;






    /*** Test Elaborazione CUDA ***/
    std::cout << "#########################" << std::endl;
    std::cout << "### Test Elaborazione ###" << std::endl;
    std::cout << "###   Versione CUDA   ###" << std::endl;
    std::cout << "#########################" << std::endl;

    ImgProcCuda imgCUDA(inputImage);

    // Determina il tipo di memoria CUDA da utilizzare
    CudaMemoryType memType = CudaMemoryType::CONSTANT_MEM;

//    cudaFree(0);  // Inizializza il runtime CUDA

    bool cudaResult;

    tstart = std::chrono::high_resolution_clock::now();
    cudaResult = imgCUDA.applyFilter( kImgFilter, memType);
    tend = std::chrono::high_resolution_clock::now();

    if (cudaResult) {
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
        std::cout << "Tempo di esecuzione CUDA: " << elapsed << " microsec" << std::endl;
        std::string outputFileImagePath = outputdir + std::string("cuda_CONSTANT_MEM_") + kImgFilter.getName()+ img_ext;
        imgCUDA.saveImageToFile(outputFileImagePath.c_str());
        std::vector<testResult> cudar;
        std::vector<testResult> testVectResultCUDA;
        testr.execTimes=elapsed;
        testr.num_iter= 1;
        testr.test_type=CUDA_CONSTANT_MEM;
        testVectResultCUDA.push_back(testr);
        testVectResult.push_back(testr);
        std::string title="Risultati test CUDA CONSTANT_MEM";
        SplashResult(title,testVectResultCUDA);
    }



    memType = CudaMemoryType::GLOBAL_MEM;

//    cudaFree(0);  // Inizializza il runtime CUDA


    tstart = std::chrono::high_resolution_clock::now();
    cudaResult = imgCUDA.applyFilter( kImgFilter, memType);
    tend = std::chrono::high_resolution_clock::now();


    if (cudaResult) {
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
        std::cout << "Tempo di esecuzione CUDA: " << elapsed << " microsec" << std::endl;
        std::string outputFileImagePath = outputdir + std::string("cuda_GLOBAL_MEM_") + kImgFilter.getName()+ img_ext;
        imgCUDA.saveImageToFile(outputFileImagePath.c_str());
        std::vector<testResult> cudar;
        std::vector<testResult> testVectResultCUDA;
        testr.execTimes=elapsed;
        testr.num_iter= 1;
        testr.test_type=CUDA_GLOBAL_MEM;
        testVectResultCUDA.push_back(testr);
        testVectResult.push_back(testr);
        std::string title="Risultati test CUDA GLOBAL_MEM";
        SplashResult(title,testVectResultCUDA);
    }

    memType = CudaMemoryType::SHARED_MEM;

//    cudaFree(0);  // Inizializza il runtime CUDA


    tstart = std::chrono::high_resolution_clock::now();
    cudaResult = imgCUDA.applyFilter( kImgFilter, memType);
    tend = std::chrono::high_resolution_clock::now();


    if (cudaResult) {
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
        std::cout << "Tempo di esecuzione CUDA: " << elapsed << " microsec" << std::endl;
        std::string outputFileImagePath = outputdir + std::string("cuda_SHARED_MEM_") + kImgFilter.getName()+ img_ext;
        imgCUDA.saveImageToFile(outputFileImagePath.c_str());
        std::vector<testResult> cudar;
        std::vector<testResult> testVectResultCUDA;
        testr.execTimes=elapsed;
        testr.num_iter= 1;
        testr.test_type=CUDA_SHARED_MEM;
        testVectResultCUDA.push_back(testr);
        testVectResult.push_back(testr);
        std::string title="Risultati test CUDA SHARED_MEM";
        SplashResult(title,testVectResultCUDA);
    }

    ImgProcOMP imgOMP(inputImage);
    int maxThreads = omp_get_max_threads();
    std::cout << "Numero massimo di thread disponibili: " << maxThreads << std::endl;

    // Test con diversi numeri di thread (potenze di 2)
    for (int numThreads = 2; numThreads <= maxThreads; numThreads *= 2) {
        tend = std::chrono::high_resolution_clock::now();
        auto ompResult = imgOMP.applyFilter(kImgFilter, numThreads);
        auto t6 = std::chrono::high_resolution_clock::now();
        if (ompResult) {
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
            std::cout << "Tempo di esecuzione OpenMP (" << numThreads << " threads): "
                      << elapsed << " microsec" << std::endl;


            std::string outputFileImagePath = outputdir + std::string("omp_") + kImgFilter.getName()+ img_ext;
            imgCUDA.saveImageToFile(outputFileImagePath.c_str());


            imgOMP.saveImageToFile(outputFileImagePath.c_str());

            std::vector<testResult> ompr;
            std::vector<testResult> testVectResultOMP;
            testr.execTimes=elapsed;
            testr.num_iter= 1;
            testr.test_type=PARALLEL;
            testVectResultOMP.push_back(testr);
            testVectResult.push_back(testr);
            std::string title="Risultati testOMP";
            SplashResult(title,testVectResultOMP);

        }
    }



    saveResultToFile(file_outPar,testVectResult);

    return 0;
}
