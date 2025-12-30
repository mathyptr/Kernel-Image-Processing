#include "util.h"
#include <fstream>
#include <filesystem> 
#include <iostream>
#include <string>
#include <regex>
#include <vector>
#include <cmath>
#include <numeric>

#include <cuda_runtime.h>


using namespace std;

void SplashScreen() {
    cout << "###############################################################"<<endl;
    cout << "###Kernel Image Processing is a set of image filtering techniques implemented through a convolution operation between an input image and a mask (kernel).###"<<endl;
    cout << "###The kernel is a small matrix that is applied to every pixel of the image: the new value of the pixel is calculated by multiplying the kernel with the surrounding pixels and summing them.###"<<endl;
    cout << "################################################################"<<endl;
}


std::string  chooseFilter() {

    cout << endl<<"######################"<< endl;
    cout << "   Scegli un filtro"<< endl;

    int filter=0;
    bool choice=false;
    while (!choice) {
        cout << "######################"<< endl;
        cout << "#1. EDGE_FILTER      #"<< endl;
        cout << "#2. SHARPEN_FILTER   #"<< endl;
        cout << "#3. GAUSSIAN_FILTER  #"<< endl;
        cout << "#4. LAPLACIAN_FILTER #"<< endl;
        cout << "#5. IDENTITY_FILTER  #"<< endl;
        cout << "######################"<< endl;
        cin >> filter;
        if (filter >= 1 && filter <=N_FILTER)
            choice=true;
        else
            cout << "Scelta non valida. Riprova di nuovo."<< endl;
    }
    std::string filter_str;
    if(filter==EDGE_FILTER)
        filter_str=EDGE_FILTER_STR;
    else if (filter==SHARPEN_FILTER)
        filter_str=SHARPEN_FILTER_STR;
    else if (filter==GAUSSIAN_FILTER)
        filter_str=GAUSSIAN_FILTER_STR;
    else if (filter==LAPLACIAN_FILTER)
        filter_str=LAPLACIAN_FILTER_STR;
    else if (filter==IDENTITY_FILTER)
        filter_str=IDENTITY_FILTER_STR;
    else
        filter_str="err";
    return filter_str;
}

int  chooseKernelSize() {

    cout << endl<<"######################"<< endl;
    cout << "   Scegli un filtro"<< endl;

    int filter_size=0;
    bool choice=false;
    while (!choice) {
        cout << "###############"<< endl;
        cout << "#1. 3x3       #"<< endl;
        cout << "#2. 5x5       #"<< endl;
        cout << "###############"<< endl;
        cin >> filter_size;
        if (filter_size >= 1 && filter_size <=N_FILTER_SIZE)
            choice=true;
        else
            cout << "Scelta non valida. Riprova di nuovo."<< endl;
    }
    int size=0;
    std::string filter_str;
    if(filter_size==size_3x3)
        size=size_3x3_value;
    else if (filter_size==size_5x5)
        size=size_5x5_value;
    else
        size=0;
    return size;
}

bool  insertFileName() {

    cout << endl;
    cout << "########################################################################"<< endl;
    cout << "Scegli se fornire un file immmagine oppure se utilizzare il file di test"<< endl;
    cout << "########################################################################"<< endl;
    bool inputfileimage=false;
    int choice=0;
    while (choice!=1 && choice!=2) {
        cout << "##################"<< endl;
        cout << "#1. Input File   #"<< endl;
        cout << "#2. File di test #"<< endl;
        cout << "##################"<< endl;
        cin >> choice;
        if (choice == 1 )
            inputfileimage=true;
        else if(choice ==2)
            inputfileimage=false;
        else
            cout << "Scelta non valida. Riprova di nuovo."<< endl;
    }
    return inputfileimage;
}



void SplashResult(string& title,std::vector<testResult>& result) {
    cout << "#####################################"<<endl;
    cout << title<< endl;

    for (const auto& res : result) {

        if( res.test_type==SEQUENTIAL)
            cout << "TEST SEQUENZIALE"<< endl;
        else if( res.test_type==PARALLEL)
            cout << "TEST PARALLELO"<< endl;
        else if( res.test_type==CUDA_CONSTANT_MEM)
            cout << "TEST CUDA"<< endl;
        else if( res.test_type==CUDA_GLOBAL_MEM)
            cout << "TEST CUDA"<< endl;
        else if( res.test_type==CUDA_SHARED_MEM)
            cout << "TEST CUDA"<< endl;
        else
            cout<<"TEST TYPE NON DEFINITO"<< endl;

        cout << "Tipo filtro: "<< res.filter_type << endl;
        cout << "Kernel size : "<< res.kernel_size << endl;
        cout << "Numero Thread : "<< res.threadNum << endl;
        cout << "Tempo di esecuzione: "<< res.execTimes<< endl;
        cout << "#####################################"<<endl;
    }
}


void saveResultToFile(const std::string& filename,std::vector<testResult>& result) {
    std::ofstream fileCSV(filename);
    fileCSV << "testType,filterType,kernelsize,numThreads,execTimes"<< endl;

    for (const auto& res : result) {
        if( res.test_type==SEQUENTIAL)
            fileCSV << "SEQUENZIALE,";
        else if( res.test_type==PARALLEL)
            fileCSV << "PARALLELO,";
        else if( res.test_type==CUDA_CONSTANT_MEM)
            fileCSV << "CUDA_CONSTANT_MEM,";
        else if( res.test_type==CUDA_GLOBAL_MEM)
            fileCSV << "CUDA_GLOBAL_MEM,";
        else if( res.test_type==CUDA_SHARED_MEM)
            fileCSV << "CUDA_SHARED_MEM,";
        else
            fileCSV<<"NON_DEFINITO,";
        fileCSV << res.filter_type <<",";
        fileCSV << res.kernel_size <<",";
        fileCSV << res.threadNum <<",";
        fileCSV << res.execTimes <<  endl;
    }
}


void checkGpuMem()
{
    float free_m,total_m,used_m;

    size_t free_t,total_t;
    cudaMemGetInfo(&free_t,&total_t);
    free_m =(uint)free_t/1048576.0 ;
    total_m=(uint)total_t/1048576.0;
    used_m=total_m-free_m;
    cout << "#####################################"<<endl;
    cout << "Mem free " << free_t << "(" << free_m << "MB)" <<endl;
    cout << "Mem total " << total_t  << "(" << total_m << "MB)" <<endl;
    cout << "Mem used " << used_m << "MB" << endl;
    cout << "#####################################"<<endl;

}