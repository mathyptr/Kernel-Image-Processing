#include "util.h"
#include <fstream>
#include <filesystem> 
#include <iostream>
#include <string>
#include <regex>
#include <vector>
#include <cmath>
#include <numeric>

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
        cout << "######################"<< endl;
        cin >> filter;
        if (filter >= 1 && filter <=4)
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
    else
        filter_str="err";
    return filter_str;
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

        cout << "Numero di Iterazioni: "<<res.num_iter<< endl;
        cout << "Tempo di esecuzione: "<<res.execTimes<< endl;
        cout << "#####################################"<<endl;
    }
}


void saveResultToFile(const std::string& filename,std::vector<testResult>& result) {
    std::ofstream fileCSV(filename);
    fileCSV << "testType,numThreads,chunkSize,execTimes,num_iter"<< endl;

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
        fileCSV << res.threadNum<<",";
        fileCSV << res.chunkSize<<",";
        fileCSV << res.execTimes<< ",";
        fileCSV << res.num_iter<< endl;
    }
}


