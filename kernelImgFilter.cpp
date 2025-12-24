#include "kernelImgFilter.h"
#include <iostream>
#include <cmath>
#include <iomanip>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


kernelImgFilter::kernelImgFilter() {
    kernelData.clear();
    size = 0;
}


bool kernelImgFilter::buildEdgeDetect() {
    std::cout << "Filtro edge detection...build..." << std::endl;
    size = 3;
    kernelData.resize(size * size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == 1 && j == 1) {
                kernelData[j + i * size] = EDGE_CENTER;
            }
            else {
                kernelData[j + i * size] = EDGE_SURROUND;
            }
        }
    }

    return true;
}


bool kernelImgFilter::buildSharp() {
    std::cout << "Filtro sharp...build..." << std::endl;
    size = 3;
    kernelData.resize(size * size);

    return Init(kernelData, SHARPEN_CENTER, SHARPEN_SURROUND, size);
}


bool kernelImgFilter::buildGaussian(int kernelSize, float sigma) {
    std::cout << "Filtro gaussiano...build..." << std::endl;

    if (kernelSize % 2 == 0 || kernelSize < 3) {
        std::cerr << "Il Kernel deve avere una dimensione dispari e maggiore di 2" << std::endl;
        return false;
    }

    if (sigma <= 0) {
        std::cerr << "Il valore di sigma deve essere positivo" << std::endl;
        return false;
    }

    size = kernelSize;
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
    size = 3;
    kernelData.resize(size * size);

    // Inizializza tutti gli elementi a -1
    for (int i = 0; i < size * size; i++) {
        kernelData[i] = LAPLACE_SURROUND;
    }

    // Imposta il valore centrale
    kernelData[4] = LAPLACE_CENTER;  // indice centrale in una matrice 3x3

    // Imposta gli angoli a 0 per ridurre la sensibilità al rumore
    kernelData[0] = 0.0f;  // alto-sinistra
    kernelData[2] = 0.0f;  // alto-destra
    kernelData[6] = 0.0f;  // basso-sinistra
    kernelData[8] = 0.0f;  // basso-destra

    return true;
}



bool kernelImgFilter::Init(std::vector<float>& kernel, float centerValue, float surroundValue, int kernelSize) {
    if (kernelSize % 2 == 0 || kernelSize < 3) {
        std::cerr << "La dimensione kernel non è valida" << std::endl;
        return false;
    }

    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            if (i == kernelSize / 2 && j == kernelSize / 2) {
                kernel[j + i * kernelSize] = centerValue;
            }
            else {
                kernel[j + i * kernelSize] = surroundValue;
            }
        }
    }

    return true;
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
        std::cout << "\n";
    }
    std::cout << "##############################" << std::endl;
}


int kernelImgFilter::getSize() const {
    return size;
}


std::vector<float> kernelImgFilter::getKernelData() const {
    return kernelData;
}
