#include "kernelImgFilter.h"
#include <iostream>
#include <cmath>
#include <iomanip>

#include "util.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


kernelImgFilter::kernelImgFilter() {
    kernelData.clear();
    size = 0;
}


std::string kernelImgFilter::getName() const{
    return name;
}

void kernelImgFilter::display() const {
    if (size == 0 || kernelData.empty()) {
        std::cout << "Il kernel non è inizializzato" << std::endl;
        return;
    }
    std::cout << "##############################" << std::endl;
    std::cout << "##### Matrice del Filtro #####\n";
    std::cout << "##############################" << std::endl;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << std::fixed << std::setprecision(4)
                      << kernelData[j + i * size] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "##############################" << std::endl;
}


int kernelImgFilter::getSize() const {
    return size;
}


std::vector<float> kernelImgFilter::getKernelData() const {
    return kernelData;
}


bool kernelImgFilter::buildFilter(std::string filter,int kernel_size) {
    name=filter;
    size=kernel_size;
    bool res=false;
    if (name == GAUSSIAN_FILTER_STR ) {
        res=buildGaussian(1.0f);  // Kernel 7x7, sigma=1.0
    }
    else if (name == SHARPEN_FILTER_STR ) {
        res=buildSharp();
    }
    else if (name == EDGE_FILTER_STR ) {
        res=buildEdgeDetect();
    }
    else if (name == LAPLACIAN_FILTER_STR ) {
        res=buildLaplacian();
    }
    else if (name == IDENTITY_FILTER_STR ) {
        res=buildIdentity();
    }
    return res;
}


bool kernelImgFilter::Init(std::vector<float>& kernel, float centerValue, float surroundValue) {
    if (size % 2 == 0 || size < 3) {
        std::cerr << "La dimensione kernel non è valida" << std::endl;
        return false;
    }
    for (auto it = begin (kernelData); it != end (kernelData); ++it)
        *it = surroundValue;
    int mid=size / 2;
    kernelData[size*size/2] = centerValue;
    return true;
}

bool kernelImgFilter::buildEdgeDetect() {
    std::cout << "Filtro edge detection...build..." << std::endl;
    kernelData.resize(size * size);
    for (auto it = begin (kernelData); it != end (kernelData); ++it)
        *it = EDGE_SURROUND;

    kernelData[size*size/2] = EDGE_CENTER;
    return true;
}

bool kernelImgFilter::buildIdentity() {
    std::cout << "Filtro Identity...build..." << std::endl;
    kernelData.resize(size * size);
    for (auto it = begin (kernelData); it != end (kernelData); ++it)
        *it = 0;
    kernelData[size*size/2] = 1;
    return true;
}

bool kernelImgFilter::buildSharp() {
    std::cout << "Filtro sharp...build..." << std::endl;
    kernelData.resize(size * size);
    return Init(kernelData, SHARPEN_CENTER, SHARPEN_SURROUND);
}


bool kernelImgFilter::buildGaussian(float sigma) {
    std::cout << "Filtro gaussiano...build..." << std::endl;

    if (size % 2 == 0 || size < 3) {
        std::cerr << "Il Kernel deve avere una dimensione dispari e maggiore di 2" << std::endl;
        return false;
    }

    if (sigma <= 0) {
        std::cerr << "Il valore di sigma deve essere positivo" << std::endl;
        return false;
    }

    kernelData.resize(size * size);
    float sum = 0.0f;

    // Calcolo centro del kernel
    int center = size / 2;

    // Creazione del filtro gaussiano
    for (int y = -center; y <= center; y++) {
        for (int x = -center; x <= center; x++) {
            // Calcolo del valore gaussiano per ogni posizione
            float value = exp(-(x * x + y * y) / (2.0f * sigma * sigma));
            value /= 2.0f * M_PI * sigma * sigma;

            kernelData[(y + center) * size + (x + center)] = value;
            sum += value;
        }
    }

    // Normalizzazione per garantire che la somma dei valori sia 1
    for (int i = 0; i < size * size; i++) {
        kernelData[i] /= sum;
    }

    return true;
}


bool kernelImgFilter::buildLaplacian() {
    std::cout << "Filtro Laplaciano...build..." << std::endl;
;
    kernelData.resize(size * size);

    // Inizializza tutti gli elementi a -1
    for (auto it = begin (kernelData); it != end (kernelData); ++it)
        *it = LAPLACE_SURROUND;

    // Imposta il valore centrale
    kernelData[size*size/2] = LAPLACE_CENTER;  // indice centrale in una matrice 3x3

    // Imposta gli angoli a 0 per ridurre la sensibilità al rumore
    kernelData[0] = 0.0f;  // alto-sinistra
    kernelData[size-1] = 0.0f;  // alto-destra
    kernelData[size*size-size] = 0.0f;  // basso-sinistra
    kernelData[size*size-1] = 0.0f;  // basso-destra

    return true;
}

