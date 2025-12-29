#ifndef UTIL_H
#define UTIL_H

#include <string>
#include <vector>

// Tipi di filtro
#define EDGE_FILTER         1   // Filtro per rilevamento bordi
#define SHARPEN_FILTER      2   // Filtro per nitidezza
#define GAUSSIAN_FILTER     3   // Filtro per sfocatura gaussiana
#define LAPLACIAN_FILTER    4   // Filtro laplaciano
#define IDENTITY_FILTER     5   // Filtro identità
#define N_FILTER    5           //Numero di filtri a disposizione
#define EDGE_FILTER_STR        "EDGE_FILTER"       // Filtro per rilevamento bordi
#define SHARPEN_FILTER_STR     "SHARPEN_FILTER"    // Filtro per nitidezza
#define GAUSSIAN_FILTER_STR    "GAUSSIAN_FILTER"   // Filtro per sfocatura gaussiana
#define LAPLACIAN_FILTER_STR   "LAPLACIAN_FILTER"  // Filtro laplaciano
#define IDENTITY_FILTER_STR    "IDENTITY_FILTER"   // Filtro identità

#define size_3x3 1
#define size_5x5 2
#define N_FILTER_SIZE 2
#define size_3x3_value 3
#define size_5x5_value 5

using namespace std;

enum testType { SEQUENTIAL, PARALLEL, CUDA_CONSTANT_MEM, CUDA_GLOBAL_MEM, CUDA_SHARED_MEM};

struct testResult {
    int threadNum=1;
    double execTimes;
    testType  test_type;
    std::string filter_type;
    int kernel_size;
};


void SplashScreen();
std::string  chooseFilter();
int  chooseKernelSize();
void SplashResult(string& title,std::vector<testResult>& result);
void saveResultToFile(const std::string& filename,std::vector<testResult>& result);
void checkGpuMem();
#endif // UTIL_H
