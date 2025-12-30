#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>

#include <omp.h>


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
    std::string imageFilePathTest = "./input/peperoni1024.png";
    std::string imageFilePath ="";
    std::string outputdir="./output/";
    std::string file_outPar = "./output/resultPAR.csv";
    std::string img_ext=".png";
    std::string fileout_suffix="";
    kernelImgFilter kImgFilter;

    std::cout << "Directory corrente: " << filesystem::current_path()  << std::endl;

    if(insertFileName()){
        cout << "Inserisci il nome file immmagine comprensivo di estensione e percorso"<< endl;
        std::cin >> imageFilePath;
    }
    else
        imageFilePath=imageFilePathTest;
    std::string  filter;
    filter=chooseFilter();
    int size=chooseKernelSize();
    kImgFilter.buildFilter(filter,size);

    // Visualizza il kernel del filtro selezionato
    std::cout << "Filtro da applicare: " << kImgFilter.getName() << std::endl;
    kImgFilter.display();


    fileout_suffix="_"+kImgFilter.getName()+std::to_string(kImgFilter.getSize())+"x"+std::to_string(kImgFilter.getSize())+img_ext;

    // Load immagine
    ImgProc inputImage;
    if (!inputImage.loadImageFromFile(imageFilePath.c_str())) {
        std::cerr << "Errore nella lettura dell'immagine sorgente: " << imageFilePath << std::endl;
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
        std::string outputFileImgaePath = outputdir + std::string("cpu") + fileout_suffix;
        imgSeq.saveImageToFile(outputFileImgaePath.c_str());
        std::vector<testResult> testVectResultSEQ;
        testr.execTimes=elapsed;
        testr.test_type=SEQUENTIAL;
        testr.filter_type=kImgFilter.getName() ;
        testr.kernel_size= kImgFilter.getSize();
        testVectResultSEQ.push_back(testr);
        testVectResult.push_back(testr);
        std::string title="Risultati test Sequenziali";
        SplashResult(title,testVectResultSEQ);
    }



    /*** Test Elaborazione CUDA ***/
    std::cout << "#########################" << std::endl;
    std::cout << "### Test Elaborazione ###" << std::endl;
    std::cout << "###   Versione CUDA   ###" << std::endl;
    std::cout << "#########################" << std::endl;


    ImgProcCuda imgCUDA(inputImage);
    checkGpuMem();
    CudaMemoryType memType = CudaMemoryType::CONSTANT_MEM;

    bool cudaResult;

    tstart = std::chrono::high_resolution_clock::now();
    cudaResult = imgCUDA.applyFilter( kImgFilter, memType);
    tend = std::chrono::high_resolution_clock::now();

    if (cudaResult) {
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
        std::string outputFileImagePath = outputdir + std::string("cuda_CONSTANT_MEM") + fileout_suffix;
        imgCUDA.saveImageToFile(outputFileImagePath.c_str());
        std::vector<testResult> cudar;
        std::vector<testResult> testVectResultCUDA;
        testr.execTimes=elapsed;
        testr.test_type=CUDA_CONSTANT_MEM;
        testr.filter_type=kImgFilter.getName() ;
        testr.kernel_size= kImgFilter.getSize();
        testVectResultCUDA.push_back(testr);
        testVectResult.push_back(testr);
        std::string title="Risultati test CUDA CONSTANT_MEM";
        SplashResult(title,testVectResultCUDA);
    }

    memType = CudaMemoryType::GLOBAL_MEM;
    tstart = std::chrono::high_resolution_clock::now();
    cudaResult = imgCUDA.applyFilter( kImgFilter, memType);
    tend = std::chrono::high_resolution_clock::now();

    if (cudaResult) {
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
        std::string outputFileImagePath = outputdir + std::string("cuda_GLOBAL_MEM") + fileout_suffix;
        imgCUDA.saveImageToFile(outputFileImagePath.c_str());
        std::vector<testResult> cudar;
        std::vector<testResult> testVectResultCUDA;
        testr.execTimes=elapsed;
        testr.test_type=CUDA_GLOBAL_MEM;
        testr.filter_type=kImgFilter.getName() ;
        testr.kernel_size= kImgFilter.getSize();
        testVectResultCUDA.push_back(testr);
        testVectResult.push_back(testr);
        std::string title="Risultati test CUDA GLOBAL_MEM";
        SplashResult(title,testVectResultCUDA);
    }

    memType = CudaMemoryType::SHARED_MEM;
    tstart = std::chrono::high_resolution_clock::now();
    cudaResult = imgCUDA.applyFilter( kImgFilter, memType);
    tend = std::chrono::high_resolution_clock::now();

    if (cudaResult) {
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
        std::string outputFileImagePath = outputdir + std::string("cuda_SHARED_MEM") + fileout_suffix;
        imgCUDA.saveImageToFile(outputFileImagePath.c_str());
        std::vector<testResult> cudar;
        std::vector<testResult> testVectResultCUDA;
        testr.execTimes=elapsed;
        testr.test_type=CUDA_SHARED_MEM;
        testr.filter_type=kImgFilter.getName() ;
        testr.kernel_size= kImgFilter.getSize();
        testVectResultCUDA.push_back(testr);
        testVectResult.push_back(testr);
        std::string title="Risultati test CUDA SHARED_MEM";
        SplashResult(title,testVectResultCUDA);
    }


    /*** Test Elaborazione OMP ***/
    std::cout << "#########################" << std::endl;
    std::cout << "### Test Elaborazione ###" << std::endl;
    std::cout << "###   Versione OMP   ###" << std::endl;
    std::cout << "#########################" << std::endl;

    std::vector<ImgProc> outputsOpenMP;
    ImgProcOMP imgOMP(inputImage);
    int maxThreads = omp_get_max_threads();
    std::cout << "Numero massimo di thread disponibili: " << maxThreads << std::endl;

    // Test con diversi numeri di thread (potenze di 2)
    for (int numThreads = 2; numThreads <= maxThreads; numThreads *= 2) {
        tstart = std::chrono::high_resolution_clock::now();
        auto ompResult = imgOMP.applyFilter(kImgFilter, numThreads);
        tend = std::chrono::high_resolution_clock::now();
        if (ompResult) {
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
            std::string outputFileImagePath = outputdir + std::string("omp_") + std::to_string(numThreads) +fileout_suffix;
            imgOMP.saveImageToFile(outputFileImagePath.c_str());
            std::vector<testResult> ompr;
            std::vector<testResult> testVectResultOMP;
            testr.execTimes=elapsed;
            testr.threadNum=numThreads;
            testr.test_type=PARALLEL;
            testr.filter_type=kImgFilter.getName() ;
            testr.kernel_size= kImgFilter.getSize();
            testVectResultOMP.push_back(testr);
            testVectResult.push_back(testr);
            std::string title="Risultati testOMP";
            SplashResult(title,testVectResultOMP);

        }
    }

    saveResultToFile(file_outPar,testVectResult);

    return 0;
}
